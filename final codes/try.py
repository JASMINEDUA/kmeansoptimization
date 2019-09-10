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

"""
Usage:
```shell

$ python workspace/classkmeans.py \
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


class kmeans(object):
    def __init__(self, name='kmeans'):
        self._ref_features = []
        self._index_to_name = []
        self._name_to_index = {}
        self._input_filenames = []
        self._feature_shape = None   
        self._feature_type = None
        self._target_root_dir = None
        self._done = False

    def compute_cosine_similarity(self,facial_features1, facial_features2):
        facial_features1 = np.array(facial_features1)
        facial_features2 = np.array(facial_features2)
        facial_features2 = np.transpose(facial_features2)
        cosine_similarity = facial_features1.dot(facial_features2)
        return (cosine_similarity)


    def compute_best_match(self,folder_names, feature):
        similarity = self.compute_cosine_similarity(self._ref_features,feature)
        output_index = np.argmax(similarity, axis=0)
        index = output_index[0][0]
        value = similarity[index]
        value = value[0][0]
        name = folder_names[index]
        return(index,name,value)

    
    def inittialize_cluster_centers(self,labeled_dir):
        counter = 0
        source_root_dir = os.path.expanduser(labeled_dir)
        class_names = os.listdir(source_root_dir)
        for class_name in class_names:                            
            class_path = os.path.join(source_root_dir,class_name)
            if not os.path.isdir(class_path):
                continue
            self._index_to_name.append(class_name)                          
            self._name_to_index[class_name] = [len(self._name_to_index)]    
            names = os.listdir(class_path)
            #print(class_name,names)
            for name in names:
                pickle_path = os.path.join(class_path,name)        
                if not os.path.isfile(pickle_path):
                    continue
                pickle_in = open(pickle_path,'rb')
                feature = pickle.load(pickle_in)
                if(self._feature_shape is None):
                    self._feature_shape = feature.shape
                    self._feature_type = feature.dtype
                    #print(self._feature.shape)
                counter += 1
                self._ref_features.append(feature)        
    #print(self._name_to_index)

    
    def generate_input_filenames(self,unlabeled_dir):
        unlabeled_root_dir = os.path.expanduser(unlabeled_dir)
        pickle_names = os.listdir(unlabeled_root_dir)
        for pickle_name in pickle_names:
            pickle_path = os.path.join(unlabeled_root_dir,pickle_name)
            if not os.path.isfile(pickle_path):
                continue
            self._input_filenames.append(pickle_path)


    def is_processed(self):
        return(self._done)


class serialkmeans(kmeans):
    def __init__(self):
        kmeans.__init__(self)
    

    def assign_cluster(self,output_dir):
        self._done = False
        renamed = False
        self._target_root_dir = os.path.expanduser(output_dir)
        for outer_index in range(len(self._input_filenames)):
            input_filename = self._input_filenames[outer_index]
            pickle_name = os.path.basename(input_filename)
            pickle_in = open(input_filename,'rb')
            feature = pickle.load(pickle_in)
            index, predicted_name, value = self.compute_best_match(self._index_to_name,feature)
            print(self._target_root_dir,predicted_name)
            target_pickle_path = os.path.join(self._target_root_dir,predicted_name)
            if (not os.path.exists(target_pickle_path)):
                os.makedirs(target_pickle_path)
            target_pickle_path = os.path.join(target_pickle_path,pickle_name)
            #print(input_filename,target_pickle_path)
            print(pickle_name, predicted_name, input_filename,target_pickle_path)
            if(input_filename != target_pickle_path):
                os.rename(input_filename, target_pickle_path)
                renamed = True
                #print('renaming file')
            self._input_filenames[outer_index] = target_pickle_path
        self._done = not renamed


    def compute_cluster_centers(self,output_dir):
        self._target_root_dir = os.path.expanduser(output_dir)
        entry_names = os.listdir(self._target_root_dir)
        for entry_name in entry_names:
            entry_path = os.path.join(self._target_root_dir,entry_name)
            if not os.path.isdir(entry_path):
                continue     
            ar = np.zeros(self._feature_shape,self._feature_type) 
            counter = 0   
            enames = os.listdir(entry_path)
            for ename in enames:
                pickle_path = os.path.join(entry_path,ename)
                if not os.path.isfile(pickle_path):
                    continue
                pickle_in = open(pickle_path,'rb')
                feature = pickle.load(pickle_in)
                ar = ar+feature
                counter += 1
            index = self._name_to_index[entry_name]
            index = index[0]
            print(index)
            ar = ar/counter
            ar = ar / LA.norm(ar)
            self._ref_features[index] = ar
            print(counter)

   
class parallelkmeans(kmeans):
    def __init__(self):
        kmeans.__init__(self)

    def assign_cluster(self,output_dir):  
        self._target_root_dir = os.path.expanduser(output_dir) 
        num_of_threads = 19    
        status = np.empty((num_of_threads), dtype=bool)
        status.fill(False)
        jobs = []
        for i in range(0, num_of_threads):
            thread = threading.Thread(target = self.list_append(status, i, num_of_threads))
            jobs.append(thread)        
    
        for j in jobs:
            j.start()
        
        for j in jobs:
            j.join()
        
        self._done = True
        for value in status:
            self._done = self._done and value


    def compute_cluster_centers(self,output_dir):
        self._target_root_dir = os.path.expanduser(output_dir)
        entry_names = os.listdir(self._target_root_dir)
        num_of_threads = 10 
        num_of_threads = min(len(entry_names), num_of_threads)
        jobs = []
        for i in range(0, num_of_threads):
            thread = threading.Thread(target = self.list_extract(entry_names, i, num_of_threads))
            jobs.append(thread) 
    
        for j in jobs:
            j.start()
        
        for j in jobs:
            j.join()     


    def list_append(self,status, i, num_of_threads):
        total_tasks = len(self._input_filenames)
        tasks_per_thread = (int(math.ceil(total_tasks / num_of_threads)))
        start = i * tasks_per_thread
        end = (i+1) * tasks_per_thread   
        if end > total_tasks:
            end = total_tasks
        
        renamed = False
        thread_done = False
    
        for outer_index in range(start, end):
            input_filename = self._input_filenames[outer_index]        
            pickle_name = os.path.basename(input_filename)
            pickle_in = open(input_filename,'rb')
            feature = pickle.load(pickle_in)
        
            index, predicted_name, value = self.compute_best_match(self._index_to_name, feature)
            target_pickle_path = os.path.join(self._target_root_dir, predicted_name)
            if (not os.path.exists(target_pickle_path)):
                os.makedirs(target_pickle_path)
            target_pickle_filename = os.path.join(target_pickle_path, pickle_name)


            if(input_filename != target_pickle_filename):
                os.rename(input_filename, target_pickle_filename)
                renamed = True
                #print('renaming file')        
                self._input_filenames[outer_index] = target_pickle_filename           
                #print(input_filename, self._input_filenames[outer_index], target_pickle_filename)        
            
        thread_done = not renamed
        status[i] = thread_done    


    def list_extract(self,entry_names, i, num_of_threads):
        whole = len(entry_names)
        tasks_per_thread = int(math.ceil(whole / num_of_threads))
        start = i * tasks_per_thread
        end = (i+1) * tasks_per_thread  
        if end > whole:
            end = whole
        
        for ind in range(start, end):
            entry_name = entry_names[ind]
            entry_path = os.path.join(self._target_root_dir,entry_name)
            print(entry_name)
            if not os.path.isdir(entry_path):
                continue 
            ar = np.zeros(self._feature_shape, self._feature_type) 
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
                index = self._name_to_index[entry_name]
                index = index[0]
                ar = ar / counter
                ar = ar / LA.norm(ar)
                self._ref_features[index] = ar



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

    #k = serialkmeans()
    k = parallelkmeans()
    k.inittialize_cluster_centers(args.labeled_dir)
    k.generate_input_filenames(args.unlabeled_dir)
    w = k.is_processed()
   

    counter = 0
    while(not w):
        k.assign_cluster(args.output_dir)
        k.compute_cluster_centers(args.output_dir)
        counter += 1
        print(counter)
        w = k.is_processed()
   


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

