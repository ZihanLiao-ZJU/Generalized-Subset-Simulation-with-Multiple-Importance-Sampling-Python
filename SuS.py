"""Subset Simulation intermediate distribution shared by both reliability toolboxes.

The driver chooses coordinates and the threshold convention explicitly. This
class does not run iterations; SeMCS owns sampling and stopping.
Both coordinate selectors store Gaussian u; they choose PF.Dist or PF.Rdis
to map u to physical performance-function inputs.
"""
import copy
import numpy as np


class SuS:
    def __init__(self, PF, coordinate_system="standard_normal",
                 threshold_rule="hazen", clip_threshold=True):
        if coordinate_system not in ("standard_normal", "physical"):
            raise ValueError("coordinate_system must be standard_normal or physical")
        if threshold_rule not in ("hazen", "midpoint"):
            raise ValueError("threshold_rule must be hazen or midpoint")
        self.TF = PF
        self.Nfun = getattr(PF, "Nfun", getattr(PF, "nfun", 1)) + 3
        self.Ndim = PF.Ndim
        self.Ncal = self.X = self.Y = None
        self.G = dict(parameters=np.array(np.inf), p=0.1, L=np.eye(self.Ndim))
        self.G["active"] = np.array(np.nan)
        self.p = 0.1
        self.Nsam = 1000
        self.coordinate_system = coordinate_system
        self.threshold_rule = threshold_rule
        self.clip_threshold = bool(clip_threshold)

    def EvlY(self, u):
        obj = copy.copy(self)
        u = np.asarray(u)
        u = u.reshape(u.shape[0], -1, order="F")
        if self.coordinate_system == "standard_normal":
            x = self.TF.Dist.U2X(u)
            log_pdf = -0.5 * (self.Ndim * np.log(2 * np.pi) + np.sum(u * u, axis=0))
        else:
            x = self.TF.Rdis.U2X(u) if hasattr(self.TF, "Rdis") else u
            log_pdf = -0.5 * (self.Ndim * np.log(2 * np.pi) + np.sum(u * u, axis=0))
        values = self.TF.EvlLSF(x) if hasattr(self.TF, "EvlLSF") else self.TF.evaluate(x)
        values = np.asarray(values, dtype=float)
        nf = self.Nfun - 3
        if nf == 1:
            values = values.reshape(1, -1)
        elif u.shape[1] == 1 and values.ndim == 1:
            values = values[:, None]
        if values.shape != (nf, u.shape[1]):
            raise ValueError("Performance function must return (Nfun, Nsam)")
        obj.X = u
        obj.Y = np.vstack([log_pdf, values])
        obj.Ncal = u.shape[1]
        return obj

    def EvlLKF(self):
        with np.errstate(divide="ignore"):
            return np.log((self.Y[1, :] < self.G["parameters"]).astype(float))

    def EvlPDF(self):
        return self.Y[0, :]

    def UpdObj(self, u, y, p=None):
        obj = copy.copy(self)
        if p is not None:
            obj.p = p
        values = np.asarray(y).reshape(np.shape(y)[0], -1, order="F")[3, :]
        if self.threshold_rule == "midpoint":
            ordered = np.sort(values)
            k = int(np.floor(len(ordered) * obj.p + 0.5))
            if k < 1 or k >= len(ordered):
                raise IndexError("Midpoint index outside sample size")
            threshold = (ordered[k - 1] + ordered[k]) / 2
        else:
            threshold = np.quantile(values, obj.p, method="hazen")
        if self.clip_threshold:
            threshold = max(threshold, 0)
        obj.G = dict(obj.G, parameters=np.array(threshold), p=obj.p)
        return y, obj.G, obj, 0

    @property
    def FlgCvg(self):
        return bool(self.G["parameters"] <= 0)

    def run_defaults(self):
        return dict(Verbose=False)

    def start_run(self,owner,sampler):
        counts=np.atleast_1d(owner.NumSam).astype(int)
        if np.any(counts<1) or owner.MaxIte<2:
            raise ValueError("NumSam must be positive and MaxIte at least 2")
        model=copy.copy(owner.IntBay)
        initial=copy.deepcopy(owner.G[0] if len(owner.G) else model.G)
        model.G=initial if isinstance(initial,dict) else dict(model.G,parameters=initial)
        if "active" not in model.G:model.G["active"]=copy.deepcopy(self.G["active"])
        sampler=copy.copy(sampler);sampler.BayStc=model
        return dict(owner=owner,model=model,sampler=sampler,limit=owner.MaxIte-1,
                    counts=counts,nc=int(counts[0]),steps=1,xseed=None,yseed=None,
                    x=[],y=[],g=[copy.deepcopy(model.G)],Nsam=[],Nsze=[],Ncal=[],selected=[],sigma=[],complete=False)

    def sample_level(self,c,level):
        if level==0: x,y,ncal,model=c["sampler"].SamGen(c["nc"],c["steps"])
        else: x,y,ncal,model=c["sampler"].SamGen(c["xseed"],c["yseed"],c["steps"])
        x=x.reshape(x.shape[0],-1,order="F");y=y.reshape(y.shape[0],-1,order="F")
        c["model"]=model
        r=dict(x=x,y=y,calls=ncal,count=c["nc"]*c["steps"],shape=(c["nc"],c["steps"]))
        c["x"].append(x);c["y"].append(y);c["Nsam"].append(r["count"]);c["Nsze"].append(r["shape"]);c["Ncal"].append(ncal)
        c["sigma"].append(getattr(c["sampler"],"sigma",None))
        return r

    def advance_level(self,c,r,level):
        model=c["model"]
        if level+1<len(c["owner"].G):
            model=copy.copy(model);previous=model.G
            specified=copy.deepcopy(c["owner"].G[level+1])
            model.G=dict(previous,**specified) if isinstance(specified,dict) else dict(previous,parameters=specified)
            for key in ("lambda","HMC_time"):
                if key in previous:model.G[key]=previous[key]
        else:
            y,_,model,extra=model.UpdObj(r["x"],r["y"],c["owner"].ConPrb)
            c["y"][-1]=y;r["y"]=y;c["Ncal"][-1]+=extra
        c["model"]=model;c["sampler"].BayStc=model;c["g"].append(copy.deepcopy(model.G))

    def select_run_seeds(self,c,r,level):
        from Support_Fun.SedSlt import SedSlt
        counts=c["counts"]
        if counts.size>1 and level+1>=counts.size:
            raise ValueError("NumSam schedule exhausted before seed selection")
        target=int(counts[0] if counts.size==1 else counts[level+1])
        xs,ys,indices=SedSlt(r["x"],r["y"],c["model"].G,target,c["sampler"].rng)
        c["xseed"],c["yseed"]=xs,ys;c["selected"].append(indices);c["nc"]=xs.shape[1]
        c["steps"]=int(np.floor(target/c["nc"]+0.5)) if c["nc"] else 0
        if c["owner"].Verbose:print(f"level {level+1}: samples={r['count']}, evaluations={c['Ncal'][-1]}")

    def run_complete(self,c,r,phase):
        if phase!="after_seeds":return False
        if c["nc"]<=0 or len(c["x"])+1==c["owner"].MaxIte:return True
        c["complete"]=bool(c["model"].FlgCvg)
        return c["complete"]

    def finish_run(self,c):
        return dict(x=c["x"],y=c["y"],g=c["g"],Nsam=np.asarray(c["Nsam"]),Nsze=np.asarray(c["Nsze"]),Ncal=np.asarray(c["Ncal"]),
                    selected=c["selected"],sigma=c["sigma"],flg_cvg=c["complete"],terminal=bool(c["model"].FlgCvg),Nite=len(c["x"]),intBay=c["model"])

