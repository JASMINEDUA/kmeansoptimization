# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
   
r"""PARAKMEANS IMPLEMENTATION.

Usage:
```shell

$ python workspace/parakmeans.py \
 	--labeled_dir /workspace/005/labeled \
        --unlabeled_dir /workspace/005/unlabeled \
	--output_dir /workspace/output 
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import numpy as np
from numpy import linalg as LA
import pickle
import math
import random
import threading


ref_features = []
index_to_name = []
name_to_index = {}


input_filenames = []
feature_shape = None
feature_type = None
target_root_dir = None
done = False


#function for computing cosine similarity
def compute_cosine_similarity(facial_features1, facial_features2):
    facial_features1 = np.array(facial_features1)
    facial_features2 = np.array(facial_features2)
    facial_features2 = np.transpose(facial_features2)
    cosine_similarity = facial_features1.dot(facial_features2)
    return (cosine_similarity)


#function for computing the best_match
def compute_best_match(folder_names,ref_features,feature):
    similarity = compute_cosine_similarity(ref_features,feature)
    output_index = np.argmax(similarity, axis=0)
    index = output_index[0][0]
    value = similarity[index]
    value = value[0][0]
    name = folder_names[index]
    return(index,name,value)


def inittialize_cluster_centers(labeled_dir):
    global index_to_name,name_to_index,ref_features
    global feature_shape, feature_type
    counter = 0
    source_root_dir = os.path.expanduser(labeled_dir)
    class_names = os.listdir(source_root_dir)
    for class_name in class_names:                            
        class_path = os.path.join(source_root_dir,class_name)
        if not os.path.isdir(class_path):
            continue
        index_to_name.append(class_name)                          
        name_to_index[class_name] = [len(name_to_index)]    
        names = os.listdir(class_path)
        #print(class_name,names)
        for name in names:
            pickle_path = os.path.join(class_path,name)        
            if not os.path.isfile(pickle_path):
                continue
            pickle_in = open(pickle_path,'rb')
            feature = pickle.load(pickle_in)
            if(feature_shape is None):
                feature_shape = feature.shape
                feature_type = feature.dtype
            counter += 1
            ref_features.append(feature)        
    #print(name_to_index)


def generate_input_filenames(unlabeled_dir):
    global input_filenames
    input_filenames = []
    unlabeled_root_dir = os.path.expanduser(unlabeled_dir)
    pickle_names = os.listdir(unlabeled_root_dir)
    for pickle_name in pickle_names:
        pickle_path = os.path.join(unlabeled_root_dir,pickle_name)
        if not os.path.isfile(pickle_path):
            continue
        input_filenames.append(pickle_path) 


def assign_cluster(output_dir):   
    global done
    global target_root_dir
    num_of_threads = 19    
    status = np.empty((num_of_threads), dtype=bool)
    status.fill(False)
    jobs = []
    for i in range(0, num_of_threads):
        thread = threading.Thread(target = list_append(status, i, num_of_threads))
        jobs.append(thread)        
    
    for j in jobs:
        j.start()
        
    for j in jobs:
        j.join()
        
    done = True
    for value in status:
        done = done and value


def compute_cluster_centers(output_dir):
    global target_root_dir 
    target_root_dir = os.path.expanduser(output_dir)
    entry_names = os.listdir(target_root_dir)
    num_of_threads = 10 
    num_of_threads = min(len(entry_names), num_of_threads)
    jobs = []
    for i in range(0, num_of_threads):
        thread = threading.Thread(target = list_extract(entry_names, i, num_of_threads))
        jobs.append(thread) 
    
    for j in jobs:
        j.start()
        
    for j in jobs:
        j.join()     



def list_append(status, i, num_of_threads):
    global input_filenames
    global ref_features, index_to_name, name_to_index
    global target_root_dir    
    
    total_tasks = len(input_filenames)
    tasks_per_thread = (int(math.ceil(total_tasks / num_of_threads)))
    start = i * tasks_per_thread
    end = (i+1) * tasks_per_thread   
    if end > total_tasks:
        end = total_tasks
        
    renamed = False
    thread_done = False
    
    for outer_index in range(start, end):
        input_filename = input_filenames[outer_index]        
        pickle_name = os.path.basename(input_filename)
        pickle_in = open(input_filename,'rb')
        feature = pickle.load(pickle_in)
        
        index, predicted_name, value = compute_best_match(index_to_name, ref_features, feature)
        target_pickle_path = os.path.join(target_root_dir, predicted_name)
        if (not os.path.exists(target_pickle_path)):
            os.makedirs(target_pickle_path)
        target_pickle_filename = os.path.join(target_pickle_path, pickle_name)


        if(input_filename != target_pickle_filename):
            os.rename(input_filename, target_pickle_filename)
            renamed = True
            #print('renaming file')        
        input_filenames[outer_index] = target_pickle_filename           
        #print(input_filename, input_filenames[outer_index], target_pickle_filename)        
            
    thread_done = not renamed
    status[i] = thread_done    


def list_extract(entry_names, i, num_of_threads):
    global feature_shape, feature_type
    global ref_features, index_to_name, name_to_index
    global target_root_dir

    whole = len(entry_names)
    tasks_per_thread = int(math.ceil(whole / num_of_threads))
    start = i * tasks_per_thread
    end = (i+1) * tasks_per_thread  
    if end > whole:
        end = whole
        
    for ind in range(start, end):
        entry_name = entry_names[ind]
        entry_path = os.path.join(target_root_dir,entry_name)
        print(entry_name)
        if not os.path.isdir(entry_path):
            continue 
        ar = np.zeros(feature_shape, feature_type) 
        counter = 0   
        enames = os.listdir(entry_path)
        for ename in enames:
            pickle_path = os.path.join(entry_path, ename)
            if not os.path.isfile(pickle_path):
                continue
            pickle_in = open(pickle_path,'rb')
            feature = pickle.load(pickle_in)
            ar = ar + feature
            counter += 1
            index = name_to_index[entry_name]
            index = index[0]
            ar = ar / counter
            ar = ar / LA.norm(ar)
            ref_features[index] = ar
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--labeled_dir',
        type=str,
        help='Input dataset image directory.',
        default='/workspace/005/labeled')

    parser.add_argument(
        '--unlabeled_dir',
        type=str,
        help='Input dataset image directory.',
        default='/workspace/005/unlabeled')

    parser.add_argument(
        '--output_dir',
        type=str,
        help=
        'Output directory where output images and data files are saved.',
        default='/workspace/output')

    return (parser.parse_args(argv))


def main(args):
    global done
    global target_root_dir 

    if (not args.labeled_dir):
        raise ValueError(
            'You must supply dataset directory with --labeled_dir.'
        )

    if (not args.unlabeled_dir):
        raise ValueError(
            'You must supply dataset directory with --unlabeled_dir.'
        )

    if (not args.output_dir):
        raise ValueError(
            'You must supply output directory for storing output data files with --output_dir.'
        )


    target_root_dir = os.path.expanduser(args.output_dir)
    if (not os.path.exists(target_root_dir)):
        os.makedirs(target_root_dir)
    inittialize_cluster_centers(args.labeled_dir)
    generate_input_filenames(args.unlabeled_dir)
    #print(input_filenames)
    counter=0 
    while(not done):
        assign_cluster(args.output_dir)
        compute_cluster_centers(args.output_dir)
        counter += 1
        print(counter)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


