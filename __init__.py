# only import if running as a custom node
try:
	pass
except ImportError:
	pass
else:
	NODE_CLASS_MAPPINGS = {}
 
 	# AnyText
	from .AnyText.nodes import NODE_CLASS_MAPPINGS as AnyText_Nodes
	NODE_CLASS_MAPPINGS.update(AnyText_Nodes)
 
	# AnyText_utils
	from .AnyText.utils import NODE_CLASS_MAPPINGS as AnyText_loader_Nodes
	NODE_CLASS_MAPPINGS.update(AnyText_loader_Nodes)
 
	NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
	__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
