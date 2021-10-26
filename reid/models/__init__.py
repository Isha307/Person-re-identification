from __future__ import absolute_import

from .resnet import *

__factory = {
    'resnet50': resnet50,
}


def create(name, *args, **kwargs):
    if name != 'resnet50':
        raise KeyError("Unknown model:", name)
    return resnet50(*args, **kwargs)
