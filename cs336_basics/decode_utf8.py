def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    print(f'bytes {list(bytestring)}')
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

def decode_utf8_bytes_to_str(bytestring: bytes):
    print(f'bytes {list(bytestring)}')
    return bytestring.decode("utf-8", errors="ignore")


# print(decode_utf8_bytes_to_str_wrong("hello 你好".encode("utf-8")))
print(decode_utf8_bytes_to_str("hello 你好".encode("utf-8")))
