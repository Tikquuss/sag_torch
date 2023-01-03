import hashlib
import os
import ntpath

def path_leaf(path):
    # https://stackoverflow.com/a/8384788/11814682
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_hash_object(type_='sha-1'):
    """make a hash object"""
    assert type_ in ["sha-1", "sha-256", "md5"]
    if type_ == 'sha-1' :
        h = hashlib.sha1()
    elif type_ == "sha-256":
        h = hashlib.sha256()
    elif type_ == "md5" :
        h = hashlib.md5()
    return h

def hash_file(file_path, BLOCK_SIZE = 65536, type_='sha-1'):
    """This function returns the SHA-1/SHA-256/md5 hash of the file passed into it
    #  BLOCK_SIZE : the size of each read from the file
    https://www.programiz.com/python-programming/examples/hash-file
    https://nitratine.net/blog/post/how-to-hash-files-in-python/
    """
    assert os.path.isfile(file_path)
    # make a hash object
    h = get_hash_object(type_)
    # open file for reading in binary mode
    with open(file_path,'rb') as file:
        # loop till the end of the file
        chunk = 0
        while chunk != b'':
            # read only BLOCK_SIZE bytes at a time
            chunk = file.read(BLOCK_SIZE)
            h.update(chunk)
    # return the hex representation of digest #, hash value as a bytes object
    return h.hexdigest() #, h.digest()

def hash_var(var, type_='sha-1'):
    """This function returns the SHA-1/SHA-256/md5 hash of the variable passed into it
    https://nitratine.net/blog/post/how-to-hash-files-in-python/
    https://stackoverflow.com/questions/24905062/how-to-hash-a-variable-in-python"""
    # make a hash object
    h = get_hash_object(type_)
    h.update(var.encode('utf8'))
    # return the hex representation of digest #, hash value as a bytes object
    return h.hexdigest() #, h.digest()

def get_hash_path(params, f, prefix, suffix) :
    f = '%s_%s'%(params.data_infos["task"], params.dataset_name) + f
    filename = "%s_%s%s"%(prefix, hash_var(f), ("_"+suffix) if suffix else "")
    data_path = os.path.join(params.log_dir, '%s.pth'%filename)
    return data_path