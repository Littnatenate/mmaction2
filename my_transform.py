import numpy as np
from mmcv.transforms import BaseTransform
from mmaction.registry import TRANSFORMS

@TRANSFORMS.register_module()
class DebugPrintShape(BaseTransform):
    """A simple transform to print the shape of the 'keypoint' tensor for debugging."""
    def __init__(self, message=""):
        self.message = message

    def transform(self, results):
        keypoint_data = results.get('keypoint')
        if keypoint_data is not None:
            # Check if it's a list of arrays or a single array
            if isinstance(keypoint_data, list):
                shapes = [item.shape for item in keypoint_data]
                print(f"DEBUG ({self.message}): Keypoint is a LIST with shapes: {shapes}")
            else:
                print(f"DEBUG ({self.message}): Keypoint shape is {keypoint_data.shape}")
        else:
            print(f"DEBUG ({self.message}): Keypoint data not found.")
        return results