# From Sompy package
# Version 1.1 modifiée LATMOS L. Barthes / Thomas Beratto 02/11/2020

import numpy as np
import sys
import inspect


class NormalizerFactory(object):

    @staticmethod
    def build(type_name):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and type_name == obj.name:
                    return obj()
        else:
            raise Exception("Unknown normalization type '%s'" % type_name)


class Normalizer(object):

    def normalize(self, data):
        raise NotImplementedError()

    def normalize_by(self, raw_data, data):
        raise NotImplementedError()

    def denormalize_by(self, raw_data, data):
        raise NotImplementedError()


class VarianceNormalizer(Normalizer):

    name = 'var'

    def _mean_and_standard_dev(self, data):
        return np.mean(data, axis=0), np.std(data, axis=0)

    def normalize(self, data):
        me, st = self._mean_and_standard_dev(data)
        #st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN #Bug à corrigé
        self.normalized = True
        self.params={'me':me,'st':st}
        return (data-me)/st                                 # modif LB ajout attribut me et st

    def normalize_by(self, raw_data, data):
        me, st = self._mean_and_standard_dev(raw_data)
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        self.normalized = True
        return (data-me)/st

    def denormalize_by(self, data_by, n_vect):
        me, st = self._mean_and_standard_dev(data_by)
        #return n_vect * st + me
        return np.squeeze(np.asarray(n_vect))*st+me                    # Modif LB
    
    def denormalize(self, data):          #ajout methode LB
        if self.normalized:
            return np.squeeze(np.asarray(data))*self.params['st']+self.params['me']
        else:
            raise 
            
    def normalize_with_params(self,data):
        return (data - self.params['me'])/self.params['st']
    


class RangeNormalizer(Normalizer):

    name = 'range'
    
    def _min_and_max(self, data):
        return min(data), max(data)
    
    def normalize(self, data):
        _min, _max = self._min_and_max(data)
        self.normalized = True
        self.params={'min':_min,'max':_max}
        return (data-_min)/(_max-_min)
    
    def denormalize(self, data):       
        if self.normalized:
            return np.squeeze(np.asarray(data))*(self.params['max']-self.params['min'])+self.params['min']
        else:
            raise 

    def normalize_with_params(self,data):
            return (data - self.params['min'])/(self.params['max']-self.params['min'])
        
    def normalize_by(self, raw_data, data):
        _min, _max = self._min_and_max(raw_data)
        self.normalized = True
        return (data-_min)/(_max-_min)

    def denormalize_by(self, data_by, n_vect):
        _min, _max = self._min_and_max(data_by)
        return np.squeeze(np.asarray(n_vect))*(_max-_min)+_min

class LogNormalizer(Normalizer):

    name = 'log'
    
    def _min(self, data):
        return min(data)
    
    def normalize(self, data):
        _min = self._min(data)
        self.normalized = True
        self.params={'min':_min}
        return np.log(data-_min+1)
    
    def denormalize(self, data):          
        if self.normalized:
            return np.squeeze(np.exp(np.asarray(data)))+self.params['min']-1
        else:
            raise 

    def normalize_with_params(self,data):
            return np.log(data - self.param['min']+1)
        
    def normalize_by(self, raw_data, data):
        _min = self._min(raw_data)
        self.normalized = True
        return np.log(data-_min+1)

    def denormalize_by(self, data_by, n_vect):
        _min = self._min(data_by)
        return np.squeeze(np.exp(np.asarray(n_vect)))+_min-1

class LogisticNormalizer(Normalizer):

    name = 'logistic'
    
    def _mean_and_standard_dev(self, data):
        return np.mean(data, axis=0), np.std(data, axis=0)

    def normalize(self, data):
        me, st = self._mean_and_standard_dev(data)
        #st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN #Bug à corrigé
        self.normalized = True
        self.params={'me':me,'st':st}
        return 1/(np.exp(-((data-me)/st))+1)                            

    def normalize_by(self, raw_data, data):
        me, st = self._mean_and_standard_dev(raw_data)
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        self.normalized = True
        return 1/(np.exp(-((data-me)/st))+1)

    def denormalize_by(self, data_by, n_vect):
        me, st = self._mean_and_standard_dev(data_by)
        return np.squeeze(-np.log(1/np.asarray(n_vect)-1))*st+me
    
    def denormalize(self, data):   
        if self.normalized:
            return np.squeeze(-np.log(1/np.asarray(data)-1))*self.params['st']+self.params['me']
        else:
            raise 
            
    def normalize_with_params(self,data):
        return 1/(np.exp(-((data-self.params['me'])/self.params['st']))+1)


class HistDNormalizer(Normalizer):

    name = 'histd'
    normalized = True


class HistCNormalizer(Normalizer):

    name = 'histc'
    normalized = True
