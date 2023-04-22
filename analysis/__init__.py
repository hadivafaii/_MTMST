from sklearnex import patch_sklearn
patch_sklearn(['PCA', 'Ridge'], verbose=False)
