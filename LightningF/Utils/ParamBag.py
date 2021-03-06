import scipy.sparse as sparse
import numpy as np
import copy


class ParamBag(object):

    ''' Container object for groups of related parameters.

    For example, can keep mean/variance params for all components
    of a Gaussian mixture model (GMM).

    Key functionality
    * run-time dimensionality verification, ensure matrices have correct size
    * easy access to the parameters for one component
    * remove/delete a particular component
    * insert new components

    Usage
    --------
    Create a new ParamBag
    >>> D = 3
    >>> PB = ParamBag(K=1, D=D)

    Add K x D field for mean parameters
    >>> PB.setField('Mu', np.ones((1,D)), dims=('K','D'))

    Add K x D x D field for all covar matrices
    >>> PB.setField('Sigma', np.eye(D)[np.newaxis,:], dims=('K','D','D'))

    >>> PB.Sigma
    array([[[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]]])

    Insert an empty component
    >>> PB.insertEmptyComps(1)

    >>> PB.K
    2
    >>> PB.Mu
    array([[ 1.,  1.,  1.],
           [ 0.,  0.,  0.]])
    '''

    def __init__(self, K=0, **kwargs):
        # type: ignore
        """ Create an object with a Bag of Parameters with specified number of components.

        Args
        --------
        K : integer number of components this bag will contain
        D : integer dimension of parameters this bag will contain
        """
        self.K = K
        if K >0:
            self.C = K - 1
        self.D = 0
        for key, val in kwargs.iteritems():
            setattr(self, key, val)
        self._FieldDims = dict()

    def copy(self):
        return copy.deepcopy(self)

    def parseArr(self, arr, dims=None, key=None):
        ''' Parse provided array-like variable into a standard numpy array
            with provided dimensions "dims", as a tuple

            Returns
            --------
            numpy array with expected dimensions
        '''
        K = self.K
        D = self.D
        if not sparse.issparse(arr):
            if not isinstance(arr, list):
                arr = np.asarray(arr, dtype=np.float64)

        # Verify shape is acceptable given expected dimensions
        if dims is not None and isinstance(dims, str):
            dims = (dims)  # force to tuple
        expectedShape = self._getExpectedShape(dims=dims)

        if arr.shape != expectedShape:
            self._raiseDimError(dims, arr, key)

        if arr.ndim == 0:
            arr = np.float64(arr)
        return arr

    def setField(self, key, rawArray, dims=None):

        # Parse dims tuple
        if dims is None and key in self._FieldDims:
            dims = self._FieldDims[key]
        else:
            self._FieldDims[key] = dims
        # Parse value as numpy array
        setattr(self, key, self.parseArr(rawArray, dims=dims, key=key))

    def insertEmptyComps(self, Kextra):
        ''' Insert Kextra empty components to self in-place.
        '''

        self.K += Kextra
        self.C += Kextra

        for key in self._FieldDims:
            dims = self._FieldDims[key]
            if dims is None:
                continue

            arr = getattr(self, key)
            for dimID, dimName in enumerate(dims):
                if dimName == 'K':
                    curShape = list(arr.shape)
                    curShape[dimID] = Kextra
                    zeroFill = np.zeros(curShape)
                    arr = np.append(arr, zeroFill, axis=dimID)
                elif dimName == 'C':
                    curShape = list(arr.shape)
                    curShape[dimID] = Kextra
                    zeroFill = np.zeros(curShape)
                    arr = np.append(arr, zeroFill, axis=dimID)

            self.setField(key, arr, dims=dims)

    def insert1Comp_fromNoise(self, idx):
        ''' Insert Kextra empty components to self in-place.
        '''

        self.K += 1
        self.C += 1

        for key in self._FieldDims:
            dims = self._FieldDims[key]
            if dims is None:
                continue

            arr = getattr(self, key)
            for dimID, dimName in enumerate(dims):
                if dimName == 'K':
                    curShape = list(arr.shape)
                    curShape[dimID] = 1
                    zeroFill = np.zeros(curShape)
                    arr = np.insert(arr, 1, np.squeeze(zeroFill), axis=dimID)
                elif dimName == 'C':
                    curShape = list(arr.shape)
                    curShape[dimID] = 1
                    zeroFill = np.zeros(curShape)
                    arr = np.insert(arr, 0, np.squeeze(zeroFill), axis=dimID)
            if key == 'rnk':
                arr[idx, 1] = copy.copy(arr[idx, 0])
                arr[idx, 0] = 0.

            self.setField(key, arr, dims=dims)

    def removeComp(self, k, noise=False):
        ''' Updates self in-place to remove component "k"
        '''

        # Validate Cluster to be erased
        if k <= 0 or k >= self.K:
            msg = 'Bad compID. Expected [1, %d], got %d' % (self.K - 1, k)
            raise IndexError(msg)
        if self.K <= 1:
            raise ValueError('Cannot remove final component.')

        # Reduce Number of Clusters
        self.K -= 1
        self.C -= 1

        # Eliminate Cluster from Params
        for key in self._FieldDims:
            arr = getattr(self, key)
            dims = self._FieldDims[key]
            if dims is not None:
                for dimID, name in enumerate(dims):
                    if name == 'K':
                        if noise and (key == 'rnk'):
                            arr[:, 0] += arr[:, k]
                        arr = np.delete(arr, k, axis=dimID)
                    if name == 'C':
                        arr = np.delete(arr, k - 1, axis=dimID)

                if key == 'rnk':
                    arr /= arr.sum(axis=1)[:, None]
                    arr[np.isnan(arr)] = 0.

                self.setField(key, arr, dims)

    def removeComps(self, k, noise=False):
        ''' Updates self in-place to remove component "k"
        '''

        # Validate Cluster to be erased
        if np.any(k <= 0) or np.any(k >= self.K):
            msg = 'Bad compID. Expected [1, %d], got %d' % (self.K - 1, k)
            raise IndexError(msg)
        if self.K <= 1:
            raise ValueError('Cannot remove final component.')

        # Reshape Clusters Correctly
        k = np.array(k)
        if k.shape == 0:
            k = np.array([k])
        k = np.squeeze(k)

        # Reduce Number of Clusters
        self.K -= len(k)
        self.C -= len(k)

        # Eliminate Cluster from Params
        for key in self._FieldDims:
            arr = getattr(self, key)
            dims = self._FieldDims[key]
            if dims is not None:
                for dimID, name in enumerate(dims):
                    if name == 'K':
                        if noise and (key == 'rnk'):
                            arr[:, 0] += arr[:, k]
                        arr = np.delete(arr, k, axis=dimID)
                    if name == 'C':
                        arr = np.delete(arr, k - 1, axis=dimID)

                if key == 'rnk':
                    arr /= arr.sum(axis=1)[:, None]
                    arr[np.isnan(arr)] = 0.

                self.setField(key, arr, dims)

    def mergeComps(self, k1, k2):
        ''' Updates self in-place to remove component "k"
        '''

        # Validate Cluster to be erased
        if k1 <= 0 or k1 >= self.K:
            msg = 'Bad compID. Expected [1, %d], got %d' % (self.K - 1, k1)
            raise IndexError(msg)
        if k2 <= 0 or k2 >= self.K:
            msg = 'Bad compID. Expected [1, %d], got %d' % (self.K - 1, k2)
            raise IndexError(msg)

        # Reduce Number of Clusters
        self.K -= 1
        self.C -= 1

        # Correctly sort comps
        if k1 > k2:
            k1, k2 = k2, k1

        # Eliminate Cluster from Params
        for key in self._FieldDims:
            arr = getattr(self, key)
            dims = self._FieldDims[key]
            if dims is not None:
                for dimID, name in enumerate(dims):
                    if name == 'K':
                        if key == 'rnk':
                            arr[:, k1] += arr[:, k2]
                        arr = np.delete(arr, k2, axis=dimID)
                    if name == 'C':
                        arr = np.delete(arr, k2 - 1, axis=dimID)

                self.setField(key, arr, dims)

    def splitComp(self, k, rnk_new):
        ''' Insert 1 empty components to self in-place.
        '''

        self.K += 1
        self.C += 1

        for key in self._FieldDims:
            dims = self._FieldDims[key]
            if dims is None:
                continue

            arr = getattr(self, key)
            for dimID, dimName in enumerate(dims):
                if dimName == 'K':
                    curShape = list(arr.shape)
                    curShape[dimID] = 1
                    zeroFill = np.zeros(curShape)
                    arr = np.insert(arr, k, np.squeeze(zeroFill), axis=dimID)
                elif dimName == 'C':
                    curShape = list(arr.shape)
                    curShape[dimID] = 1
                    zeroFill = np.zeros(curShape)
                    arr = np.insert(arr, k - 1, np.squeeze(zeroFill), axis=dimID)

            if key == 'rnk':
                arr[:, k:k+2] = rnk_new

            self.setField(key, arr, dims=dims)

    def _getExpectedShape(self, key=None, dims=None):
        """
        Returns tuple of expected shape, given named dimensions.
        :param key:
        :param dims:
        :return:
        """

        if key is not None:
            dims = self._FieldDims[key]

        if dims is None:
            expectShape = ()
        else:
            shapeList = list()
            for dim in dims:
                if isinstance(dim, int):
                    shapeList.append(dim)
                else:
                    shapeList.append(getattr(self, dim))
            expectShape = tuple(shapeList)

        return expectShape

    def _raiseDimError(self, dims, badArr, key=None):
        ''' Raise ValueError when expected dimensions for array are not met.
        '''
        expectShape = self._getExpectedShape(dims=dims)
        if key is None:
            msg = 'Bad Dims. Expected %s, got %s' % (expectShape, badArr.shape)
        else:
            msg = 'Bad Dims for field %s. Expected %s, got %s' % (
                key, expectShape, badArr.shape)

        if sparse.issparse(badArr):
            msg = msg + ". Array is SPARSE. If expected array is 1D, this cannot be met by sparse array."

        raise ValueError(msg)

    # ######
    def reorderComps(self, sortIDs):
        ''' Rearrange internal order of all fields along dimension 'K'
        '''
        for key in self._FieldDims:
            arr = getattr(self, key)
            dims = self._FieldDims[key]
            if arr.ndim == 0:
                continue
            if dims[0] == 'K' and 'K' not in dims[1:]:
                arr = arr[sortIDs]
            elif dims[0] == 'K' and dims[1] == 'K' and 'K' not in dims[2:]:
                arr = arr[sortIDs, :][:, sortIDs]
            elif 'K' not in dims:
                continue
            elif dims[0] != 'K' and dims[1] == 'K':
                arr = arr[:, sortIDs]
            elif dims[0] != 'K' and dims[2] == 'K':
                arr = arr[:, :, sortIDs]
            else:
                raise NotImplementedError('TODO' + key + str(dims))
            self.setField(key, arr, dims=dims)

    def setComp(self, k, compPB):
        ''' Set (in-place) component k of self to provided compPB object.
        '''
        if k < 0 or k >= self.K:
            emsg = 'Bad compID. Expected [0, %d] but provided %d'
            emsg = emsg % (self.K - 1, k)
            raise IndexError(emsg)
        if compPB.K != 1:
            raise ValueError('Expected compPB to have K=1')
        for key, dims in self._FieldDims.items():
            if dims is None:
                self.setField(key, getattr(compPB, key), dims=None)
            elif self.K == 1:
                self.setField(key, getattr(compPB, key), dims=dims)
            else:
                bigArr = getattr(self, key)
                bigArr[k] = getattr(compPB, key)  # in-place

    def getComp(self, k, doCollapseK1=True):
        ''' Returns ParamBag object for component "k" of self.
        '''
        if k < 0 or k >= self.K:
            emsg = 'Bad compID. Expected [0, %d] but provided %d'
            emsg = emsg % (self.K - 1, k)
            raise IndexError(emsg)
        cPB = ParamBag(K=1, D=self.D, doCollapseK1=doCollapseK1)
        for key in self._FieldDims:
            arr = getattr(self, key)
            dims = self._FieldDims[key]
            if dims is not None:
                if self.K == 1:
                    cPB.setField(key, arr, dims=dims)
                else:
                    singleArr = arr[k]
                    if doCollapseK1:
                        cPB.setField(key, singleArr, dims=dims)
                    elif singleArr.ndim == 0:
                        cPB.setField(key, singleArr[np.newaxis], dims=dims)
                    else:
                        cPB.setField(key, singleArr[np.newaxis, :], dims=dims)
            else:
                cPB.setField(key, arr)
        return cPB

    def __add__(self, PB):
        ''' Add. Returns new ParamBag, with fields equal to self + PB
        '''
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        PBsum = ParamBag(K=self.K, D=self.D, doCollapseK1=self.doCollapseK1)
        for key in self._FieldDims:
            arrA = getattr(self, key)
            arrB = getattr(PB, key)
            PBsum.setField(key, arrA + arrB, dims=self._FieldDims[key])
        return PBsum

    def __iadd__(self, PB):
        ''' In-place add. Updates self, with fields equal to self + PB.
        '''
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        if len(self._FieldDims.keys()) < len(PB._FieldDims.keys()):
            for key in PB._FieldDims:
                arrB = getattr(PB, key)
                try:
                    arrA = getattr(self, key)
                    self.setField(key, arrA + arrB)
                except AttributeError:
                    self.setField(key, arrB.copy(), dims=PB._FieldDims[key])
        else:
            for key in self._FieldDims:
                arrA = getattr(self, key)
                arrB = getattr(PB, key)
                self.setField(key, arrA + arrB)

        return self

    def subtractSpecificComps(self, PB, compIDs):
        ''' Subtract (in-place) from self the entire bag PB
                self.Fields[compIDs] -= PB
        '''
        assert len(compIDs) == PB.K
        for key in self._FieldDims:
            arr = getattr(self, key)
            if arr.ndim > 0:
                arr[compIDs] -= getattr(PB, key)
            else:
                self.setField(key, arr - getattr(PB, key), dims=None)

    def __sub__(self, PB):
        ''' Subtract.

        Returns new ParamBag object with fields equal to self - PB.
        '''
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        PBdiff = ParamBag(K=self.K, D=self.D, doCollapseK1=self.doCollapseK1)
        for key in self._FieldDims:
            arrA = getattr(self, key)
            arrB = getattr(PB, key)
            PBdiff.setField(key, arrA - arrB, dims=self._FieldDims[key])
        return PBdiff

    def __isub__(self, PB):
        ''' In-place subtract. Updates self, with fields equal to self - PB.
        '''
        if self.K != PB.K or self.D != PB.D:
            raise ValueError('Dimension mismatch')
        for key in self._FieldDims:
            arrA = getattr(self, key)
            arrB = getattr(PB, key)
            self.setField(key, arrA - arrB)
        return self
