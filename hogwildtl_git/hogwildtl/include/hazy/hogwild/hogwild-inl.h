// Copyright 2012 Chris Re, Victor Bittorf
//
 //Licensed under the Apache License, Version 2.0 (the "License");
 //you may not use this file except in compliance with the License.
 //You may obtain a copy of the License at
 //    http://www.apache.org/licenses/LICENSE-2.0
 //Unless required by applicable law or agreed to in writing, software
 //distributed under the License is distributed on an "AS IS" BASIS,
 //WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 //See the License for the specific language governing permissions and
 //limitations under the License.

// The Hazy Project, http://research.cs.wisc.edu/hazy/
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)

#ifndef HAZY_HOGWILD_HOGWILD_INL_H
#define HAZY_HOGWILD_HOGWILD_INL_H
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <map>
#include <sys/mman.h>
#include <cstdio>
#include <boost/filesystem.hpp>
#include "hazy/util/clock.h"
#include "hazy/hogwild/freeforall-inl.h"
#include "../../../../src/svm/svmmodel.h"
#include <omp.h>
#include <cassert>
#include <iomanip>
// See for documentation
#include "hazy/hogwild/hogwild.h"
char *szExampleFile, prob_file[40];
int* random_seq, degree, sample_count;
volatile int train_ended;
volatile double variance=0;

std::vector<std::string> split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

__inline__ void swap (int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void randomize ( int arr[], int n, int seed )
{
    // Use a different seed value so that we don't get same
    // result each time we run this program
    srand ( time(NULL)+seed );
 
    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (int i = n-1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        int j = rand() % (i+1);
 
        // Swap arr[i] with the element at random index
        swap(&arr[i], &arr[j]);
    }
}


namespace hazy {
namespace hogwild {
	
namespace svm {
	vector::FVector<SVMExample> train_examps;

    void Release_IS_Sequence(void){
        for(int i=0;i<nthreads;i++){
        		delete[] IS_seq[i]; 
				delete[] IS_dataset[i];
                //delete[] grad_hist[i];
        }
        delete[] random_seq;
        delete[] train_examps.values;
    }
    
	void init_thread_dataset(void){
        int len=train_examps.size/nthreads;
        int left=train_examps.size%nthreads;
        int start,end;
        sample_count = train_examps.size;
        for(int i=0;i<nthreads;i++){
            start=i*len;
            end=len*(i+1)+(i==(nthreads-1))*left;
            len=end-start;
            int *seq= new int[len];
			int *dataset= new int[len];
            for(int count=0;count<len;count++){
              seq[count]=0;
			  dataset[count]=0;
            }
			IS_dataset.push_back(dataset);
			IS_seq.push_back(seq);
        }
		
		random_seq=new int[train_examps.size];
		for(int i=0;i<train_examps.size;i++){
			random_seq[i]=i;
		}
		//std::cout<<"Init thread dataset finished\n";
	}

    void Generate_Thread_Dataset(void){
        int len=train_examps.size/nthreads;
        int left=train_examps.size%nthreads;
        int start,end;
        int *dataset;
        for(int i=0;i<nthreads;i++){
            start=i*len;
            end=len*(i+1)+(i==(nthreads-1))*left;
            len=end-start;
            dataset= IS_dataset[i];
            for(int count=0;count<len;count++){
              dataset[count]=random_seq[start+count];
            }
        }
		//std::cout<<"Generate thread dataset finished\n";
        return;  
    }

  void Construct_IS_Distribution(void) {
    int len=train_examps.size;
    double prob_total=0.0,prob_tmp=0.0;    
    int *seq,idx;
    seq=random_seq;
    prob_tmp=0.0;
    prob_total=0.0;
    for(int j = 0;j<len;j++){
        idx=seq[j];
        train_examps[idx].p_s=prob_tmp;
        train_examps[idx].p_e=train_examps[idx].p_s+train_examps[idx].probability;
        train_examps[idx].update_weight=train_examps[idx].probability*len;
        prob_tmp+=train_examps[idx].probability;
        if(use_prob==0){
          train_examps[idx].update_weight=1;
        }
    }
    //std::cout<<"Rescaling prob finished\n";
    return;
  }

  void Generate_Sample_Sequence_All(int epoch) {
        int len=train_examps.size/nthreads;  
        srand (epoch);
        int *seq, *dataset;
        double rand_num;
        int sel_idx;
        int data_idx,left,right;
        bool found;
        dataset = random_seq;
        //std::ofstream rmse_output;
        //rmse_output.open(seqfile,std::ios_base::out);
        //std::cout<<"-----------------start-----------------\n";
        for(int i=0;i<nthreads;i++){
            int start =i*len;
            seq = IS_seq[i];
            int trunk_len=len+(i==(nthreads-1))*(train_examps.size%nthreads);
            for(int count=0;count<trunk_len;count++){
                rand_num=(double)rand()/(RAND_MAX);
                left=0,right=train_examps.size;
                found = 0;
                while(left<=right){
                    data_idx=(left+right)/2;
                    sel_idx=dataset[data_idx];
                    if((rand_num>train_examps[sel_idx].p_s)&&(rand_num<=train_examps[sel_idx].p_e)){
                        found = 1;
                        break;
                    } else if(rand_num>train_examps[sel_idx].p_e){
                        left = data_idx+1;
                    } else {
                        right = data_idx-1;
                    }
                }
                seq[count]=sel_idx;

            }
        }
        //std::cout<<"-----------------end-----------------\n";
        return;
    }


  void Generate_Sample_Sequence_Fast(int epoch) {
      int len=train_examps.size/nthreads;  
      srand (time(NULL));
      int *seq, *dataset;
      double rand_num;
      int sel_idx;
      int data_idx,left,right;
      bool found;
      //std::ofstream rmse_output;
      //rmse_output.open(seqfile,std::ios_base::out);
      //std::cout<<"-----------------start-----------------\n";
      for(int i=0;i<nthreads;i++){
          int start =i*len;
          seq = IS_seq[i];
          dataset = IS_dataset[i];
          int trunk_len=len+(i==(nthreads-1))*(train_examps.size%nthreads);
          for(int count=0;count<trunk_len;count++){
             // if(use_prob){
                  rand_num=(double)rand()/(RAND_MAX);
                  left=0,right=trunk_len;
                  found = 0;
                  while(left<=right){
                      data_idx=(left+right)/2;
                      sel_idx=dataset[data_idx];
                      if((rand_num>train_examps[sel_idx].p_s)&&(rand_num<=train_examps[sel_idx].p_e)){
                          found = 1;
                          break;
                      } else if(rand_num>train_examps[sel_idx].p_e){
                          left = data_idx+1;
                      } else {
                          right = data_idx-1;
                      }
                  }
                  assert(found==1);
                  //std::cout<<"sel_idx "<<sel_idx<<std::endl;
                  seq[count]=sel_idx;
                  if(use_prob==0){
                    seq[count]=dataset[rand()%trunk_len];
                  }
              //} else {
              //    seq[count]=dataset[rand()%trunk_len];
                  //seq[count]=random_seq[start+count];
              //}
          }
      }
      //std::cout<<"-----------------end-----------------\n";
      return;
  }

}

template <class Model, class Params, class Exec>
template <class Scan>
void Hogwild<Model, Params, Exec>::UpdateModel(Scan &scan, int epoch) {
    scan.Reset();
    static int gen=0;
    
    if(gen==0){
        Generate_Thread_Dataset();
        //RescaleProbability();
        Construct_IS_Distribution();
        gen=1;
    }
    Generate_Sample_Sequence_All(epoch);
    //Shuffle_Sample_Sequence();
    Zero();
    train_time_.Start();
    epoch_time_.Start();  
	if(svrg){
	// for gradient variance evaluation, we have to calculate optimal gradient
	    model_.SnapShot();
	    //Calculate_Full_Gradient_Parallel(svm::train_examps,params_,&model_);
	    Calculate_Full_Gradient_Serial(svm::train_examps,params_,&model_);
	}
    FFAScan(model_, params_, scan, tpool_, Exec::UpdateModel, res_);
    epoch_time_.Stop();
    train_time_.Pause();  
}

template <class Model, class Params, class Exec>
template <class Scan>
double Hogwild<Model, Params, Exec>::ComputeRMSE(Scan &scan) {
  scan.Reset();
  Zero();
  test_time_.Start();
  size_t count = FFAScan(model_, params_, scan,tpool_, Exec::TestModel, res_);
  test_time_.Stop();

  double sum_sqerr = 0;
  for (unsigned i = 0; i < tpool_.ThreadCount(); i++) {
    sum_sqerr += res_.values[i];
  }
  return std::sqrt(sum_sqerr) / std::sqrt(count);
}


template <class Model, class Params, class Exec>
template <class TrainScan, class TestScan>
void Hogwild<Model, Params, Exec>::RunExperiment(
    int nepochs, hazy::util::Clock &wall_clock, TrainScan &trscan, TestScan &tescan) {
  printf("wall_clock: %.5f    Going Hogwild!\n", wall_clock.Read());
  std::ofstream rmse_output;
  std::string fname="_rmse_";
  std::string fname1(szExampleFile);
  double epoch_grad,sum_grad_thread,variance_thread;
  const size_t last_slash_idx = fname1.find_last_of("\\/");
  if (std::string::npos != last_slash_idx)
  {
      fname1.erase(0, last_slash_idx + 1);
  }
  double train_rmse;
  rmse_output.open(fname1+fname+std::to_string(tpool_.ThreadCount()),std::ios_base::out);
  train_rmse = ComputeRMSE(trscan);
  std::cout<<"init train rmse "<<train_rmse<<std::endl;  
  //train_rmse = ComputeRMSE(tescan);
  //std::cout<<"init test rmse "<<train_rmse<<std::endl;

  for (int e = 1; e <= nepochs; e++) {
    Model* m_cpy=model_.Clone();
    UpdateModel(trscan,e);
    train_rmse = ComputeRMSE(trscan);
    double test_rmse = 0;
    epoch_grad=0;
    sum_grad_thread=0;
    variance_thread=0;
    double const * const /*__restrict__*/ uvals = model_.weights.values;
    double const * const /*__restrict__*/ vvals = m_cpy->weights .values;
    #if 1
    for (int i = model_.weights.size; i-- > 0; ) {
      epoch_grad += (uvals[i] - vvals[i])*(uvals[i] - vvals[i]); 
      //epoch_grad += (uvals[i] - vvals[i]); 
    }
    //epoch_grad=sqrt(epoch_grad);
    #endif
    
    delete m_cpy;  
    variance_thread=((epoch_grad-model_.grad_norm)*(epoch_grad-model_.grad_norm));
    
    test_rmse = ComputeRMSE(tescan);
	Exec::PostEpoch(model_, params_);
    printf("epoch: %d wall_clock: %.5f train_time: %.5f test_time: %.5f epoch_time: %.5f train_rmse: %.5f test_rmse: %.5f epoch_grad: %.10f variance: %.10f\n", 
           e, wall_clock.Read(), train_time_.value, test_time_.value, 
           epoch_time_.value, train_rmse, test_rmse, epoch_grad, variance_thread);
    fflush(stdout);
	rmse_output<<std::setprecision(9)<<train_rmse<<" "<<test_rmse<<" "<<train_time_.value<<" "<<epoch_grad<<" "<<variance_thread<<std::endl;
	
  }
  rmse_output.close();
}

} // namespace hogwild
} // namespace hazy

#endif


