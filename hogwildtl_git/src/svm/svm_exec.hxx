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
#include <iostream>
#include <fstream>
#include <cmath>  
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#include "hazy/hogwild/tools-inl.h"

namespace hazy {
namespace hogwild {
namespace svm {
fp_type inline ComputeLoss(const SVMExample &e, const SVMModel& model) {
  // determine how far off our model is for this example
  vector::FVector<fp_type> const &w = model.weights;
  fp_type dot = vector::Dot(w, e.vector);
  double loss = std::max((1 - dot * e.value), static_cast<fp_type>(0.0));
  return loss;
}

void inline ModelUpdateSVRG(const SVMExample &examp, const SVMParams &params, SVMModel *model,vector::FVector<fp_type> *tmp) {
  vector::FVector<fp_type> &w = model->weights;
  vector::FVector<fp_type> &ss = model->weights_snapshot;
  vector::FVector<fp_type> &avgG = model->avg_gradients;

//--------------------------------------
//-------- calculate gardient of snapshot

  fp_type wxys = vector::Dot(ss, examp.vector);
  wxys = wxys * examp.value;
    for (unsigned i = tmp->size; i-- > 0; ) {
	    tmp->values[i] = 0;
  }  
  
  if (wxys < 1) {
	  fp_type e = -2*(1-wxys)*params.step_size*examp.value;
  	  vector::ScaleAndAdd(*tmp, examp.vector, e);
  }

  fp_type * const vals_tmp = tmp->values;
  unsigned const * const degs_tmp = params.degrees;
  size_t const size_tmp = examp.vector.size;
  // update based on the evaluation
  fp_type const scalar_tmp = (params.step_size*params.mu);
  for (int d = size_tmp; d-- > 0; ) {
  	int const j = examp.vector.index[d];
  	unsigned const deg = degs_tmp[j];
  	vals_tmp[j] += scalar_tmp / deg;
  }

//-------------------------------

  // evaluate this example
  fp_type wxy = vector::Dot(w, examp.vector);
  wxy = wxy * examp.value;

  if (wxy < 1) { // hinge is active.
    fp_type e = 2*(1-wxy)*params.step_size * examp.value;
    vector::ScaleAndAdd(*tmp, examp.vector, e);
  }
  
  fp_type * const vals = tmp->values;
  size_t const size = examp.vector.size;
  unsigned const * const degs = params.degrees;

  // update based on the evaluation
  fp_type const scalar = (params.step_size * params.mu);
  for (int i = size; i-- > 0; ) {
    int const j = examp.vector.index[i];
    unsigned const deg = degs[j];
    vals[j] -= scalar / deg;
  }

  vector::ScaleAndAdd(*tmp, avgG, -1.0);
	vector::ScaleAndAdd(w, *tmp, 1.0);	

}

void inline ModelUpdateSVRG_Quick(const SVMExample &examp, const SVMParams &params, SVMModel *model) {
  vector::FVector<fp_type> &w = model->weights;
  vector::FVector<fp_type> &ss = model->weights_snapshot;
  vector::FVector<fp_type> &avgG = model->avg_gradients;

//-------------------------------

  // evaluate this example
  fp_type wxy = vector::Dot(w, examp.vector);
  wxy = wxy * examp.value;

  if (wxy < 1) { // hinge is active.
    fp_type e = 2*(1-wxy)*params.step_size * examp.value;
    vector::ScaleAndAdd(w, examp.vector, e);
  }
  
  fp_type * const vals = w.values;
  size_t const size = examp.vector.size;
  unsigned const * const degs = params.degrees;

  // update based on the evaluation
  fp_type const scalar = (params.step_size * params.mu);
  for (int i = size; i-- > 0; ) {
    int const j = examp.vector.index[i];
    unsigned const deg = degs[j];
    vals[j] -= scalar / deg;
  }

//--------------------------------------
//-------- calculate gardient of snapshot

  fp_type wxys = vector::Dot(ss, examp.vector);
  wxys = wxys * examp.value;
  
  if (wxys < 1) {
	  fp_type e = -2*(1-wxys)*params.step_size*examp.value;
  	  vector::ScaleAndAdd(w, examp.vector, e);
  }

  fp_type * const vals_tmp = w.values;
  unsigned const * const degs_tmp = params.degrees;
  size_t const size_tmp = examp.vector.size;
  // update based on the evaluation
  fp_type const scalar_tmp = (params.step_size*params.mu);
  for (int d = size_tmp; d-- > 0; ) {
  	int const j = examp.vector.index[d];
  	unsigned const deg = degs_tmp[j];
  	vals_tmp[j] += scalar_tmp / deg;
  }

	vector::ScaleAndAdd(w, avgG, -1.0);

}


#define LINEAR 0
void inline ModelUpdate(const SVMExample &examp, const SVMParams &params, SVMModel *model) {
  vector::FVector<fp_type> &w = model->weights;
  //SVMModel* m_cpy=model->Clone();
  
  // evaluate this example
  fp_type wxy = vector::Dot(w, examp.vector);

  wxy = wxy * examp.value;

  if (wxy < 1) { // hinge is active.
    fp_type const e = 2*(1-wxy)*(params.step_size * examp.value)/examp.update_weight;
    vector::ScaleAndAdd(w, examp.vector, e);
  }

    fp_type * const vals = w.values;
    unsigned const * const degs = params.degrees;
    size_t const size = examp.vector.size;
      
    // update based on the evaluation
    fp_type const scalar = params.step_size * params.mu/examp.update_weight;
    for (int i = size; i-- > 0; ) {
      int const j = examp.vector.index[i];
      unsigned const deg = degs[j];
      vals[j] -= scalar / deg;
    }
}

void SVMExec::PostUpdate(SVMModel &model, SVMParams &params) {
  // Reduce the step size to encourage convergence
  params.step_size *= params.step_decay;
}

void SVMExec::PostEpoch(SVMModel &model, SVMParams &params) {
  // Reduce the step size to encourage convergence
  params.step_size *= params.step_decay;
}

double SVMExec::UpdateModel(SVMTask &task, unsigned tid, unsigned total) {
  SVMModel  &model = *task.model;
  SVMParams const &params = *task.params;
  vector::FVector<SVMExample> const & exampsvec = task.block->ex;
  // calculate which chunk of examples we work on
  size_t start = hogwild::GetStartIndex(exampsvec.size, tid, total); 
  size_t end = hogwild::GetEndIndex(exampsvec.size, tid, total);
  // optimize for const pointers 
  size_t *perm = task.block->perm.values;
  SVMExample const * const examps = exampsvec.values;
  SVMModel * const m = &model;
  int idx;
  double rand_num;
  int *seq;
  seq=IS_seq[tid];
  vector::FVector<fp_type> tmp; 
  //std::cout<<"tid "<<tid<<" update model start\n";
  if(svrg){
    tmp.size = m->weights.size;
    tmp.values = new fp_type[tmp.size];
  }
  //std::cout<<"tid "<<tid<<" update model with length "<<end-start<<std::endl;
  for (unsigned i = 0; i < end-start; i++) {
    if(svrg==0){
      ModelUpdate(examps[seq[i]], params, m);
    }else{
      //ModelUpdateSVRG_Quick(examps[seq[i]], params, m);
      ModelUpdateSVRG(examps[seq[i]], params, m,&tmp);
//    if((i%1000==0)&&(tid==0)){
//        std::cout<<"svrg "<<i<<" finished\n";
//    }      
    }
  }
  if(svrg){
    delete tmp.values;
  }
  //std::cout<<"update model fin\n";
  return 0.0;
}

double SVMExec::TestModel(SVMTask &task, unsigned tid, unsigned total) {
  SVMModel const &model = *task.model;

  //SVMParams const &params = *task.params;
  vector::FVector<SVMExample> const & exampsvec = task.block->ex;

  // calculate which chunk of examples we work on
  size_t start = hogwild::GetStartIndex(exampsvec.size, tid, total); 
  size_t end = hogwild::GetEndIndex(exampsvec.size, tid, total);

  // keep const correctness
  SVMExample const * const examps = exampsvec.values;
  fp_type loss = 0.0;
  // compute the loss for each example
  for (unsigned i = start; i < end; i++) {
    fp_type l = ComputeLoss(examps[i], model);
    loss += l;
  }
  //std::cout<<"svm test model\n";
  // return the number of examples we used and the sum of the loss
  //counted = end-start;
  return loss;
}

} // namespace svm
} // namespace hogwild

} // namespace hazy
