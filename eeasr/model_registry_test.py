from eeasr import model_registry

@model_registry.RegisterSingleTaskModel
class DummyModel(object):
  def __init__():
    pass


print(model_registry.GetAllRegisteredClasses())
