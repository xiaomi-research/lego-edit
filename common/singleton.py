class Singleton(object):
    _INSTANCE = {}
    def __init__(self, cls):
        self.cls = cls
         
    def __call__(self, *args, **kwargs):
        instance = self._INSTANCE.get(self.cls, None)
        if not instance:
            instance = self.cls(*args, **kwargs)
            self._INSTANCE[self.cls] = instance
        return instance
     
    def __getattr__(self, key):
        return getattr(self.cls, key, None)
 

    pass