// Copyright 2012 Victor Bittorf, Chris Re
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Hogwild!, part of the Hazy Project
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)
// Original Hogwild! Author: Chris Re (chrisre [at] cs.wisc.edu)             

#ifndef HAZY_HOGWILD_INSTANCES_SVM_MODEL_H
#define HAZY_HOGWILD_INSTANCES_SVM_MODEL_H

#include "hazy/vector/fvector.h"
#include "hazy/vector/svector.h"
#include "hazy/vector/operations-inl.h"
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"

#include "hazy/hogwild/hogwild_task.h"
#include <mutex>
#include <pthread.h>


namespace hazy {
namespace hogwild {
//! Sparse SVM implementation
namespace svm {

std::mutex mtx;

//! The precision of the values, either float or double
typedef double fp_type;
struct SVMParams {
  float mu; //!< mu param
  float step_size; //!< stepsize (decayed by step decay at each epoch)
  float step_decay; //!< factor to modify step_size by each epoch
  unsigned const *degrees; //!< degree of each feature
  unsigned ndim; //!< number of features, length of degrees

  //! Constructs a enw set of params
  SVMParams(fp_type stepsize, fp_type stepdecay, fp_type _mu) :
      mu(_mu), step_size(stepsize), step_decay(stepdecay) { }
};

//! A single example which is a value/rating and a vector
struct SVMExample {
  fp_type value; //!< rating of this example
  vector::SVector<const fp_type> vector; //!< feature vector
  double probability;
  double update_weight;
  double p_s;
  double p_e;
  SVMExample() { }
  //! Constructs a new example
  /*! Makes an example backed by the given memory.
   * \param val example's rating
   * \param values the values of the features (for a sprase vector)
   * \param index the indicies of the values
   * \param len the length of index and values
   */
  SVMExample(fp_type val, fp_type const * values, int * index, 
             unsigned len, double prob) :
      value(val), vector(values, index, len), probability(prob){ }

  SVMExample(const SVMExample &o) {
    value = o.value;
    vector.values = o.vector.values;
    vector.index = o.vector.index;
    vector.size = o.vector.size;
  }
};

//! The mutable model for a sparse SVM.
struct SVMModel {
  //! The weight vector that is trained
  vector::FVector<fp_type> weights;
  vector::FVector<fp_type> weights_snapshot;
  vector::FVector<fp_type> avg_gradients;
  vector::FVector<fp_type> tmp;
  double grad_norm;
  //! Construct a weight vector of length dim backed by the buffer
  /*! A new model backed by the buffer.
   * \param buf the backing memory for the weight vector
   * \param dim the length of buf
   */
  explicit SVMModel(unsigned dim) {
    weights.values = new fp_type[dim];
	  weights_snapshot.values = new fp_type[dim];
	  avg_gradients.values = new fp_type[dim];
	  tmp.values = new fp_type[dim];
      weights.size = dim;
      tmp.size = dim;
	  avg_gradients.size = dim;
    for (unsigned i = dim; i-- > 0; ) {
        weights.values[i] = 0;
	      avg_gradients.values[i] = 0;
	      tmp.values[i] = 0;
    }
  }

  void SnapShot(void){
    for (unsigned i = weights.size; i-- > 0; ) {
        weights_snapshot.values[i]=weights.values[i];
    }	
  }
  void Reset_Avg_Gradient(){
	  for (unsigned i = avg_gradients.size; i-- > 0; ) {
		    avg_gradients.values[i] = 0;
	  }
  }
  /*! Copies from the given model into this model.
   * This is required for the 'Best Ball' training.
   */
  void CopyFrom(SVMModel const &m) {
    assert(weights.size == m.weights.size);
    vector::CopyInto(m.weights, weights);
  }

  /*! Creates a deep copy of this model, caller must free.
   * This is required for the 'Best Ball' training.
   */
  SVMModel* Clone() {
    SVMModel *m = new SVMModel(weights.size);
    m->CopyFrom(*this);
    return m;
  }

  void SyncModel(SVMModel* const m_dest, SVMModel* const m_src, int tid){
      int dim = weights.size;
      printf("tid %d sync\n",tid);
      mtx.lock();
      for(int i=dim;i-->0;){
        m_dest->weights.values[i]=m_dest->weights.values[i]-(m_dest->weights.values[i]-m_src->weights.values[i])/2.0;
      }
      mtx.unlock();
  }

};

//! Parameters for SVM training

  #define NUM_THREADS (32)
    struct thread_data{
     unsigned int tid;
     const vector::FVector<SVMExample> *examp;
     const SVMParams *params;
     SVMModel *model;
     vector::FVector<fp_type> local_grad;
  };


  void* Kernel_Calculate_LocalFullGradient(void *threadarg){ 
      struct thread_data *kernel_data;
      kernel_data = (struct thread_data *) threadarg;
      int tid=kernel_data->tid;
      //std::cout<<"tid "<<tid<<std::endl;
      vector::FVector<fp_type> &w = kernel_data->model->weights;
      for (unsigned d = kernel_data->local_grad.size; d-- > 0; ) {
        kernel_data->local_grad.values[d] = 0;
      } 
      fp_type * const vals = kernel_data->local_grad.values;
      unsigned const * const degs = kernel_data->params->degrees; 
      fp_type const scalar = (kernel_data->params->step_size*kernel_data->params->mu);
      int idx=0;
      int count=kernel_data->examp->size/NUM_THREADS+(tid==(NUM_THREADS-1))*kernel_data->examp->size%NUM_THREADS;
      int start=(kernel_data->examp->size/NUM_THREADS)*tid;
      //std::cout<<"start "<<start<<" end "<<start+count<<std::endl;
      for(int i=start;i<start+count;i++){
          // evaluate this example
          fp_type wxy = vector::Dot(w, kernel_data->examp->values[i].vector);
          wxy = wxy * kernel_data->examp->values[i].value;
          //std::cout<<"tmp size "<<tmp.size<<"w size "<<w.size<<" "<<avgGrad.size<<std::endl;
          if (wxy < 1) { // hinge is active.
            fp_type const e = -2*(1-wxy)*kernel_data->params->step_size*kernel_data->examp->values[i].value;
            vector::ScaleAndAdd(kernel_data->local_grad, kernel_data->examp->values[i].vector, e);
          }     
          // update based on the evaluation   
          size_t const size = kernel_data->examp->values[i].vector.size;
          for (int d = size; d-- > 0; ) {
            int const j = kernel_data->examp->values[i].vector.index[d];
            unsigned const deg = degs[j];
            vals[j] += scalar / deg;
          }   
          if(idx++%100000==0){
             //std::cout<<"tid: "<<tid<<" "<<float(idx)/count<<" finished\n";
          }
      }
        size_t const size = w.size;
        for (int d = size; d-- > 0; ) {
          vals[d]/=kernel_data->examp->size;   
        }
    pthread_exit(NULL);
  }
  


  void Calculate_Full_Gradient_Parallel(const vector::FVector<SVMExample> &examp, const SVMParams &params, SVMModel *model) {
    //clear full gradient
    model->Reset_Avg_Gradient();
    vector::FVector<fp_type> &w = model->weights;
    vector::FVector<fp_type> &avgGrad = model->avg_gradients;
    pthread_t threads[NUM_THREADS];
    struct thread_data td[NUM_THREADS];
    int rc;
    void *status;
    for(int i=0; i < NUM_THREADS; i++ ){
        //std::cout << "PFGD : creating thread, " << i <<" with w.size as "<<w.size<< std::endl;
        td[i].tid=i;
        td[i].examp = &examp;
        td[i].params = &params;
        td[i].model = model;
        td[i].local_grad.values = new fp_type[w.size];
        td[i].local_grad.size = w.size;
        rc = pthread_create(&threads[i], NULL, Kernel_Calculate_LocalFullGradient, (void *)&td[i]);
        if (rc){
            std::cout << "Error:unable to create thread," << rc << std::endl;
            exit(-1);
        }     
        
    }
    for(int i=0; i < NUM_THREADS; i++ ){
        rc = pthread_join(threads[i], &status);
        if (rc){
            std::cout << "Error:unable to join," << rc << std::endl;
            exit(-1);
        }
        //std::cout<<"tid "<<i<<" joined \n";
    }   
    // aggregating the gradients
    size_t const size = w.size;
    for(int i=0; i < NUM_THREADS; i++ ){
        for (int d = size; d-- > 0; ) {
          avgGrad.values[d]+=td[i].local_grad.values[d];   
        } 
        delete td[i].local_grad.values;
    } 

  
  }



  void Calculate_Full_Gradient_Serial(const vector::FVector<SVMExample> &examp, const SVMParams &params, SVMModel *model) {
  model->Reset_Avg_Gradient();
  vector::FVector<fp_type> &w = model->weights;
  vector::FVector<fp_type> &avgGrad = model->avg_gradients;
  vector::FVector<fp_type> &tmp= model->tmp; 
  for(int i=0;i<examp.size;i++){
	  // evaluate this example
	  fp_type wxy = vector::Dot(w, examp.values[i].vector);
	  wxy = wxy * examp.values[i].value;
	  //std::cout<<"tmp size "<<tmp.size<<"w size "<<w.size<<" "<<avgGrad.size<<std::endl;
	  for (unsigned k = tmp.size; k-- > 0; ) {
		tmp.values[k] = 0;
	  } 

	  if (wxy < 1) { // hinge is active.
	    fp_type const e = -2*(1-wxy)*params.step_size*examp.values[i].value;
	    vector::ScaleAndAdd(tmp, examp.values[i].vector, e);
	  }

	  fp_type * const vals = tmp.values;
	  unsigned const * const degs = params.degrees;
	  size_t const size = examp.values[i].vector.size;
	
	  // update based on the evaluation
	  fp_type const scalar = (params.step_size*params.mu);
	  for (int d = size; d-- > 0; ) {
		int const j = examp.values[i].vector.index[d];
		unsigned const deg = degs[j];
		vals[j] += scalar / deg;
	  }       
	    vector::ScaleAndAdd(avgGrad, tmp, 1.0);
  }


  	size_t const size = w.size;
    for (int d = size; d-- > 0; ) {
  	  avgGrad.values[d]/=examp.size;   
    } 

}

typedef HogwildTask<SVMModel, SVMParams, SVMExample> SVMTask;

} // namespace svm
} // namespace hogwild

} // namespace hazy

#endif
