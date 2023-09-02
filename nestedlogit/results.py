import numpy as np
import pandas as pd
import scipy.stats
import warnings


class ModelResults:
    def __init__(self, model, params, cov_params):
        self.model = model
        self.params = params
        self.cov_params = cov_params

        self.llf = model.loglike(params)
        self.nobs = model.nobs
        self.df_resid = model.df_resid
        self.num_params = model.num_params
        self.aic = 2 * (self.num_params - self.llf)
        self.bic = self.nobs * np.log(self.num_params) - 2 * self.llf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.bse = np.sqrt(np.diag(self.cov_params))
            self.tvalues = self.params / self.bse
            self.pvalues = scipy.stats.norm.sf(np.abs(self.tvalues)) * 2

        self.null_results = None

    def use_fit_null(self, *args, **kwargs):
        """Calculates the stuff based on the results of model.fit_null"""
        self.null_results = self.model.fit_null(*args, **kwargs)
        self.llnull = self.null_results.llf
        self.llr = 2 * (self.llf - self.llnull)
        df = self.model.num_params - self.null_results.model.num_params
        self.llr_pvalue = scipy.stats.chi2.sf(self.llr, df)
        self.prsquared = 1 - self.llf / self.llnull

    def conf_int(self, alpha=0.05):
        q = scipy.stats.norm.ppf(1 - alpha / 2)
        lower = self.params - q * self.bse
        upper = self.params + q * self.bse
        return np.hstack((lower[:, None], upper[:, None]))

    def summary(self, alpha=0.05):
        model_name = self.model.__class__.__name__
        toret = [model_name + " Model Results"]
        toret.append("-" * len(toret[0]))

        info_list = [
            ("Model:", model_name),
            ("No. Observations:", self.nobs),
            ("Df Residuals:", self.df_resid),
            ("Df Model:", self.num_params),
            ("Log-Likelihood:", "%8.5g" % self.llf),
        ]
        if self.null_results is not None:
            null_res_info = [
                ("Pseudo R-squ.:", "%#6.4g" % self.prsquared),
                ("LL-Null:", "%#8.5g" % self.llnull),
                ("LLR p-value:", "%#6.4g" % self.llr_pvalue),
            ]
            info_list.extend(null_res_info)
        info_df = pd.DataFrame(info_list)
        toret.append(info_df.to_string(header=False, index=False))
        toret.append(toret[1])

        param_info_cols = [
            "",
            "value",
            "std err",
            "z",
            "P>|z|",
            f"[{alpha / 2}",
            f"{1 - alpha / 2}]",
        ]
        pnames = self.model.param_names
        cint = self.conf_int(alpha=alpha)
        param_info = [
            (
                name,
                "%#10.4f" % val,
                "%#9.3f" % stderr,
                "%#9.3f" % tval,
                "%#9.3f" % pval,
                "%#9.3f" % lo,
                "%#9.3f" % hi,
            )
            for name, val, stderr, tval, pval, (lo, hi) in zip(
                pnames, self.params, self.bse, self.tvalues, self.pvalues, cint
            )
        ]
        param_info_df = pd.DataFrame(param_info, columns=param_info_cols)
        toret.append(param_info_df.to_string(index=False))
        toret.append(toret[1])

        if self.null_results is None:
            toret.append(
                "Call use_fit_null to get prsquared, llnull, and llr_pvalue."
            )

        return "\n".join(toret)
