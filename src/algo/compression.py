import blosc

class DummyCompressor(object):
    def compress_elem(self, elem):
        return elem
    def decompress_elem(self, elem):
        return elem

class BloscArrayCompressor(DummyCompressor):
    def compress_elem(self, elem):
        return blosc.pack_array(elem)
    def decompress_elem(self, elem):
        return blosc.unpack_array(elem)

if __name__ == "__main__":
    pass