import blosc

import timeit
from algo.neural_architecture import basic_transition

import numpy as np
import torch

import cv2

class DummyCompressor(object):
    def compress_transition(self, transition):
        return transition
    def decompress_transition(self, transition):
        return transition

class TransitionCompressor(object):
    def compress_elem(self, elem):
        return blosc.pack_array(elem)
    def decompress_elem(self, elem):
        return blosc.unpack_array(elem)
    def compress_transition(self, transition):
        """
        Expected type is a torch.tensor, convert it to numpy here
        """
        compressed_transitions = basic_transition(current_state=self.compress_elem(transition.current_state.numpy()),
                                                  action=transition.action,
                                                  reward=transition.reward,
                                                  next_state=self.compress_elem(transition.next_state.numpy()),
                                                  terminal=transition.terminal,
                                                  mission=transition.mission,
                                                  mission_length=transition.mission_length,
                                                  gamma=transition.gamma
                                                  )

        return compressed_transitions

    def decompress_transition(self, transition):

        compressed_transitions = basic_transition(current_state=torch.Tensor(self.decompress_elem(transition.current_state)),
                                                  action=transition.action,
                                                  reward=transition.reward,
                                                  next_state=torch.Tensor(self.decompress_elem(transition.next_state)),
                                                  terminal=transition.terminal,
                                                  mission=transition.mission,
                                                  mission_length=transition.mission_length,
                                                  gamma=transition.gamma
                                                  )

        return compressed_transitions

class TransitionCompressorPNG(TransitionCompressor):
    def compress_elem(self, elem):
        elem = elem.squeeze(0).swapaxes(0,2).swapaxes(0,1)
        _, im = cv2.imencode('.png', elem)
        return im
    def decompress_elem(self, elem):
        im = cv2.imdecode(elem, cv2.IMREAD_COLOR)
        elem = np.expand_dims(im.swapaxes(1,0).swapaxes(2,0), 0)
        return elem


class TransitionCompressorTest(TransitionCompressor):
    def __init__(self, config, array_size):
        super().__init__()
        self.typesize = config["typesize"]
        self.cname = config["cname"]
        self.array_mode = config["array_mode"] if "array_mode" in config else False
        self.array_size = array_size

    def compress_elem(self, elem):
        if self.array_mode:
            compressed = blosc.pack_array(elem)
        else:
            compressed = blosc.compress(elem.astype(np.float).tostring(), typesize=self.typesize, cname=self.cname)
        return compressed

    def decompress_elem(self, elem):
        if self.array_mode:
            decompressed = blosc.unpack_array(elem)
        else:
            decompressed = np.fromstring(blosc.decompress(elem)).reshape(self.array_size)
        return decompressed

if __name__ == "__main__":

    import sys
    import torch

    def get_size(obj, seen=None):
        """Recursively finds size of objects"""

        if isinstance(obj, torch.Tensor):
            return obj.nelement() * obj.element_size()

        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size


    array_size = (128,128,3)
    number = 1000
    image = torch.Tensor(np.load("test_image.npy"))

    compress1 = TransitionCompressorTest(config=dict([('typesize', 1), ("cname", 'lz4')]), array_size=array_size)
    compress2 = TransitionCompressorTest(config=dict([('typesize', 8), ("cname", 'lz4')]), array_size=array_size)

    compress3 = TransitionCompressorTest(config=dict([('typesize', 1), ("cname", 'blosclz')]), array_size=array_size) # Close to lz4, but ratio is lower
    compress4 = TransitionCompressorTest(config=dict([('typesize', 8), ("cname", 'blosclz')]), array_size=array_size)

    compress5 = TransitionCompressorTest(config=dict([('typesize', 1), ("cname", 'blosclz'), ("array_mode", True)]), array_size=array_size) # Superior in speed !

    compress6 = TransitionCompressorTest(config=dict([('typesize', 4), ("cname", 'zlib')]), array_size=array_size) # Compression time is loooong
    compress7 = TransitionCompressorTest(config=dict([('typesize', 1), ("cname", 'zlib')]), array_size=array_size) # Compression time is loooong

    compress8 = TransitionCompressorPNG()


    t = basic_transition(current_state=image,
                         next_state=image,
                         action=0,
                         reward=1,
                         terminal=2,
                         mission_length=3,
                         mission=4,
                         gamma=5)


    print("Size :")
    original_size = get_size(t.current_state)
    print("Original size ", original_size)
    print(get_size(compress1.compress_transition(t).current_state) / original_size)
    print(get_size(compress3.compress_transition(t).current_state) / original_size)
    print(get_size(compress5.compress_transition(t).current_state) / original_size)
    print(get_size(compress6.compress_transition(t).current_state) / original_size)
    print(get_size(compress8.compress_transition(t).current_state) / original_size)



    print("Compression time :")
    print(timeit.timeit(lambda : compress1.compress_transition(t), number=number))
    print(timeit.timeit(lambda : compress2.compress_transition(t), number=number))
    print(timeit.timeit(lambda: compress3.compress_transition(t), number=number))
    print(timeit.timeit(lambda: compress4.compress_transition(t), number=number))
    print(timeit.timeit(lambda: compress5.compress_transition(t), number=number))
    print(timeit.timeit(lambda: compress8.compress_transition(t), number=number))

    t_compress = compress1.compress_transition(t)
    t_compress_array = compress5.compress_transition(t)
    t_png = compress8.compress_transition(t)

    print("Decompression time :")
    print(timeit.timeit(lambda : compress1.decompress_transition(t_compress), number=number))
    print(timeit.timeit(lambda : compress2.decompress_transition(t_compress), number=number))
    print(timeit.timeit(lambda : compress3.decompress_transition(t_compress), number=number))
    print(timeit.timeit(lambda : compress4.decompress_transition(t_compress), number=number))
    print(timeit.timeit(lambda : compress5.decompress_transition(t_compress_array), number=number))
    print(timeit.timeit(lambda : compress8.decompress_transition(t_png), number=number))

    a = compress5.decompress_transition(t_compress_array)
    #print(a.current_state.shape)