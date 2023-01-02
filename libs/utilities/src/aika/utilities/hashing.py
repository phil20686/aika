import typing as t
import hashlib


def session_consistent_hash(obj: t.Any, hash_object=None):
    """
    This should be able to take any of the parameters that go into making a metadata object,
    it will essentially function as the hash function for metadata objects which must be
    consistent across python sessions. Python randomises its string hashes.
    """
    if hash_object is None:
        hash_object = hashlib.md5()
    print(hash_object.digest_size)
    print(hash_object.block_size)
    if isinstance(obj, str):
        hash_object.update(obj.encode("utf16"))
    elif isinstance(obj, t.Mapping):
        for key, value in obj.items():
            session_consistent_hash(key, hash_object=hash_object)
            session_consistent_hash(value, hash_object=hash_object)
    elif isinstance(obj, t.Sequence):
        for value in obj:
            session_consistent_hash(value, hash_object=hash_object)
    elif isinstance(obj, t.FrozenSet):
        # the use of hash_object=None will guarantee an order independent
        # of the session, and we can then use that order to update the final hash.
        session_consistent_hash(
            list(
                sorted(
                    ["%x" % session_consistent_hash(x, hash_object=None) for x in obj]
                )
            ),
            hash_object=hash_object,
        )
    else:
        hash_object.update(("%x" % hash(obj)).encode("utf16"))
    # we restrict to standard signed 64 bit sizes as this is typically the max in of various
    # database containers like eg mongo ints.
    return int(hash_object.hexdigest(), 16) % 9223372036854775807
