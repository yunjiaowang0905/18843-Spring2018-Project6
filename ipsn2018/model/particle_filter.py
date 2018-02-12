class particle_filter(object):
    """Particle Filter
    This class offers the training and prediction of Particle Filter
    Parameters
    ---------
    n_mix: int, optional
        The number of mixture components of the GMM
        Default set to 32.
    n_iter: int, optional
        THe number of iteration for EM algorithm.
        Default set to 100.
    covtype: str, optional
        The type of covariance matrix of the GMM
        full: full-covariance matrix
        block_diag (not implemeted) : block-diagonal matrix
    Attributes
    ---------
    param :
        Sklean-based model parameters of the GMM
    """

    def __init__(self, conf):
        self.conf = conf
        pass

    def update(self):
        pass

    def predict(self):
        pass
