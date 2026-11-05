"""GSS-MIS numerical implementation.

Xia and Liao (2025), Enhanced generalized subset simulation with multiple
importance sampling for reliability estimation, Computers & Structures 313,
107741. https://doi.org/10.1016/j.compstruc.2025.107741

Generalized subset simulation: Li, Ma and Cao (2015), Computers &
Structures 153, 239-251, doi:10.1016/j.compstruc.2014.10.014.

Arrays store coordinates/functions by row and samples by column.
"""

import copy
import numpy as np


def support(f, g, h):
    g = g["parameters"] if isinstance(g, dict) else g
    g = np.broadcast_to(np.asarray(g), (f.shape[0],))
    h = np.broadcast_to(np.asarray(h, dtype=bool), (f.shape[0],))
    return np.any((f < g[:, None])[h], axis=0)


def likelihood(f, g, h):
    return np.where(support(f, g, h), 0.0, -np.inf)


def update_y(y, g, h):
    yy = y.copy()
    yy[0] = likelihood(yy[3:], g, h)
    yy[1] = yy[2]
    return yy


class GSS_MIS:
    """Union intermediate distribution; iteration is owned by SeMCS."""
    def __init__(self, PF):
        self.TF = PF
        self.Nfun = getattr(PF, "Nfun", getattr(PF, "nfun", 1)) + 3
        self.Ndim = PF.Ndim
        self.Ncal = self.X = self.Y = None
        self.G = dict(parameters=np.array(np.inf), p=0.1, L=np.eye(self.Ndim))
        self.G["active"] = np.array(True)
        self.p = 0.1
        self.Nsam = 1000

    def EvlY(self, x):
        obj = copy.copy(self)
        obj.X = x
        physical = self.TF.Rdis.U2X(x) if hasattr(self.TF, "Rdis") else x
        values = self.TF.EvlLSF(physical) if hasattr(self.TF, "EvlLSF") else self.TF.evaluate(physical)
        obj.Y = np.vstack((-0.5 * (self.Ndim * np.log(2 * np.pi) + np.sum(x * x, axis=0)), values))
        obj.Ncal = x.shape[1]
        return obj

    def EvlLKF(self):
        return likelihood(self.Y[1:], self.G, self.G["active"])

    def EvlPDF(self):
        return self.Y[0]

    def UpdObj(self, x, y, p=None):
        if p is not None:
            self.p = p
        from Support_Fun.numerics import quantile
        self.G = dict(self.G, parameters=np.maximum(quantile(y[3:], self.p), 0), p=self.p)
        self.G["active"] = self.G["active"] & (self.G["parameters"] > 0)
        return y, self.G, self, 0

    @property
    def FlgCvg(self):
        return not np.any(self.G["active"])

    def run_defaults(self):
        return dict(Verbose=False)

    def start_run(self, owner, sampler):
        counts = np.atleast_1d(owner.NumSam).astype(int)
        if np.any(counts < 1) or owner.MaxIte < 2:
            raise ValueError("NumSam must be positive and MaxIte at least 2")
        model = copy.copy(owner.IntBay)
        initial = copy.deepcopy(owner.G[0] if len(owner.G) else model.G)
        model.G = initial if isinstance(initial, dict) else dict(model.G, parameters=initial)
        if "active" not in model.G:
            model.G["active"] = copy.deepcopy(self.G["active"])
        sampler = copy.copy(sampler)
        sampler.BayStc = model
        return dict(
            owner=owner,
            model=model,
            sampler=sampler,
            limit=owner.MaxIte - 1,
            counts=counts,
            nc=int(counts[0]),
            steps=1,
            xseed=None,
            yseed=None,
            x=[],
            y=[],
            g=[copy.deepcopy(model.G)],
            Nsam=[],
            Nsze=[],
            Ncal=[],
            selected=[],
            sigma=[],
            complete=False,
        )

    def sample_level(self, context, level):
        if level == 0:
            x, y, ncal, model = context["sampler"].SamGen(context["nc"], context["steps"])
        else:
            x, y, ncal, model = context["sampler"].SamGen(
                context["xseed"], context["yseed"], context["steps"]
            )
        x = x.reshape(x.shape[0], -1, order="F")
        y = y.reshape(y.shape[0], -1, order="F")
        context["model"] = model
        record = dict(
            x=x,
            y=y,
            calls=ncal,
            count=context["nc"] * context["steps"],
            shape=(context["nc"], context["steps"]),
        )
        context["x"].append(x)
        context["y"].append(y)
        context["Nsam"].append(record["count"])
        context["Nsze"].append(record["shape"])
        context["Ncal"].append(ncal)
        context["sigma"].append(getattr(context["sampler"], "sigma", None))
        return record

    def advance_level(self, context, record, level):
        model = context["model"]
        if level + 1 < len(context["owner"].G):
            model = copy.copy(model)
            previous = model.G
            specified = copy.deepcopy(context["owner"].G[level + 1])
            model.G = (
                dict(previous, **specified)
                if isinstance(specified, dict)
                else dict(previous, parameters=specified)
            )
            for key in ("lambda", "HMC_time"):
                if key in previous:
                    model.G[key] = previous[key]
        else:
            y, _, model, extra = model.UpdObj(record["x"], record["y"], context["owner"].ConPrb)
            context["y"][-1] = y
            record["y"] = y
            context["Ncal"][-1] += extra
        context["model"] = model
        context["sampler"].BayStc = model
        context["g"].append(copy.deepcopy(model.G))

    def select_run_seeds(self, context, record, level):
        from Support_Fun.SedSlt import SedSlt

        counts = context["counts"]
        if counts.size > 1 and level + 1 >= counts.size:
            raise ValueError("NumSam schedule exhausted before seed selection")
        target = int(counts[0] if counts.size == 1 else counts[level + 1])
        xs, ys, indices = SedSlt(
            record["x"], record["y"], context["model"].G, target, context["sampler"].rng
        )
        context["xseed"], context["yseed"] = xs, ys
        context["selected"].append(indices)
        context["nc"] = xs.shape[1]
        context["steps"] = int(np.floor(target / context["nc"] + 0.5)) if context["nc"] else 0
        if context["owner"].Verbose:
            print(f"level {level+1}: samples={record['count']}, evaluations={context['Ncal'][-1]}")

    def run_complete(self, context, record, phase):
        if phase != "after_seeds":
            return False
        if context["nc"] <= 0 or len(context["x"]) + 1 == context["owner"].MaxIte:
            return True
        context["complete"] = bool(context["model"].FlgCvg)
        return context["complete"]

    def finish_run(self, context):
        return dict(
            x=context["x"],
            y=context["y"],
            g=context["g"],
            Nsam=np.asarray(context["Nsam"]),
            Nsze=np.asarray(context["Nsze"]),
            Ncal=np.asarray(context["Ncal"]),
            selected=context["selected"],
            sigma=context["sigma"],
            flg_cvg=context["complete"],
            terminal=bool(context["model"].FlgCvg),
            Nite=len(context["x"]),
            intBay=context["model"],
        )
