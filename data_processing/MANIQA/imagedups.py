#!python
import argparse
from PIL import Image
import os
import sys
import imagehash
import progressbar
import multiprocessing as mp
import numpy as np
import cv2
def dupes(config):
    hmap = {}
    paths = config['paths']
    subdirs = []
    if config['recurse']:
        for path in paths:
            for root, dirs, _ in os.walk(path):
                for name in dirs:
                    subdirs.append(os.path.join(root, name))
    paths += subdirs
    files = []
    for path in paths:
        fs = os.listdir(path)
        for f in fs:
            fpath = os.path.join(path, f)
            if os.path.isdir(fpath):
                continue
            files.append(fpath)
    
    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()
    results = [pool.apply_async(async_hash, args=(fpath, managed_dict, managed_locker)) for fpath in files]
    
    pbar = progressbar.ProgressBar(max_value=len(files))
    for i, p in enumerate(results):
        p.get()
        pbar.update(i)
    pbar.finish()

    count = 0
    for k, v in managed_dict.items():
        if len(v) == 1:
            continue

        # show image in v
        if config['show']:
            images = []
            for fpath in v:
                images.append(Image.open(fpath))
            images = [np.array(image) for image in images]
            images = np.concatenate(images, axis=1)
            images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
            images = cv2.resize(images, (images.shape[1] // 4, images.shape[0] // 4))
            cv2.imshow('images', images)
            cv2.waitKey(0)

        for idx, fpath in enumerate(v):
            if idx == 0:
                if not config['quiet']:
                    #print("[+]", fpath, os.path.getsize(fpath))
                    pass
            else:
                if not config['quiet']:
                    pass
                    #print("[-]", fpath, os.path.getsize(fpath))

                confirm = config['noprompt']



                if not config['noprompt'] and config['delete']:
                    print("Delete %s? [y/n]")
                    confirm = sys.stdin.readline().strip() == 'y'
                if config['delete'] and confirm:
                    count += 1
                    os.unlink(fpath)
        # if not config['quiet']:
        #     print()


    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Deleted %d files" % count)
def async_hash(fpath, result_dict, result_lock):
    try:
        h = imagehash.average_hash(Image.open(fpath))
        h = "%s" % h
        sims = result_dict.get(h, [])
        sims.append(fpath)
        with result_lock:
            result_dict[h] = sims
    except Exception as e:
        pass

def main(args=None):
    parser = argparse.ArgumentParser(
        prog="imagedups",
        description="""Find/Delete duplicated images
    
  imagedups [options] -p DIRECTORY...
        """,
        epilog="""
  inspire by fdupes
    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('-d', '--delete', dest='delete', default=False, action='store_true', 
                            help='Delete duplicated files, keep one image only')
    parser.add_argument('-r', '--recurse', dest='recurse', default=False, action='store_true', 
                            help='For every directory given follow subdirectories encountered within')
    parser.add_argument('-N', '--noprompt', dest='noprompt', default=False, action='store_true', 
                            help='''Together with --delete, preserve the first file in
each set of duplicates and delete the rest without
prompting the user
                            ''')
    parser.add_argument('-w', '--show', dest='show', default=False, action='store_true',
                        help='''Together with --delete, preserve the first file in
    each set of duplicates and delete the rest without
    prompting the user
                                ''')
    parser.add_argument('-q', '--quiet', dest='quiet', default=False, action='store_true', 
                            help='Hide progress indicator')
    parser.add_argument('--minsize', dest='minsize', type=int,
                            help='Consider only files greater than or equal to SIZE bytes')
    parser.add_argument('--maxsize', dest='maxsize', type=int,
                            help='Consider only files less than or equal to SIZE bytes')
    parser.add_argument('-p', '--path', dest='paths', nargs='+', type=str, required=True)

    if args is not None:
        config = vars(parser.parse_args(args))
    else:
        config = vars(parser.parse_args())
    
    dupes(config)
    
if __name__ == '__main__':
    main()