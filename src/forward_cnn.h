#ifndef FORWARD_CNN_H
#define FORWARD_CNN_H

#include <vector>
#include <cassert>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>


template<typename T>
class Tensor{
public:
    std::vector<T> data;
    std::vector<int> shape;
    int n;

    // constructors
    Tensor() {}

    Tensor(int a){
        shape = std::vector<int>(1);
        shape[0] = a;
        n = a;
        data = std::vector<T>(n, 0);
    }

    Tensor(int a, int b){
        shape = std::vector<int>(2);
        shape[0] = a;
        shape[1] = b;
        n = a * b;
        data = std::vector<T>(n, 0);
    }

    Tensor(std::vector<int>& s) {
        shape = s;
        n = 1;
        for(int i=0;i<s.size();i++){
            n *= s[i];
        }
        data = std::vector<T>(n, 0);
    }

    Tensor(std::vector<T>& d, std::vector<int>& s){
        assert(s.size() > 0);
        data = d;
        shape = s;
        n = 1;
        for(int i=0;i<s.size();i++){
            n *= s[i];
        }
        assert(n == data.size());
    }

    // getters
    T& at(int a){
        assert(shape.size() == 1);
        assert(0 <= a && a < shape[0]);
        return data[a];
    }

    T& at(int a, int b){
        assert(shape.size() == 2);
        assert(0 <= a && a < shape[0] && 0 <= b && b < shape[1]);
        return data[a * shape[1] + b];
    }

    T& at(std::vector<int>& pos){
        assert(pos.size() == shape.size());
        int p = 0;
        for(int i=0;i<pos.size();i++){
            assert(0 <= pos[i] && pos[i] < shape[i]);
            p *= shape[i];
            p += pos[i];
        }
        return data[p];
    }

    T& ix(int pos){
        assert(0 <= pos && pos < n);
        return data[pos];
    }

    // utility
    void reshape(std::vector<int> s){
        int n_ = 1;
        for(int i=0;i<s.size();i++){
            n_ *= s[i];
        }
        assert(n == n_);
        shape = s;
    }
};


class layer {
public:
    Tensor<float> output;

    virtual ~layer() {}
    virtual void build() {}
    virtual void forward(Tensor<float>& input) {}
    virtual std::vector<int> get_input_shape() {return input_shape;}
    virtual std::vector<int> get_output_shape() {return output_shape;}
    virtual void set_input_shape(std::vector<int> shape) {input_shape = shape;}
    virtual void load_weights(std::istringstream& ss){}

    virtual void print_weights() {}
protected:
    std::vector<int> input_shape, output_shape;
};


class Dense: public layer {
public:
    Dense(int output_dim, int input_dim=0){
        output_shape.resize(1);
        output_shape[0] = output_dim;
        if(input_dim > 0){
            input_shape.resize(1);
            input_shape[0] = input_dim;
        }else{
            input_shape.clear();
        }
    }

    virtual void build(){
        assert(input_shape.size() == 1 && input_shape[0] > 0);
        assert(input_shape.size() == 1 && input_shape[0] > 0);
        n_in = input_shape[0];
        n_out = output_shape[0];
        // This is a placeholder.
        // if you train the network, you should init this randomly.
        W = Tensor<float>(n_in, n_out);
        b = Tensor<float>(n_out);
    }

    virtual void forward(Tensor<float>& input){
        assert(input.shape.size() == 2 && input.shape[1] == n_in);
        int n = input.shape[0];
        output = Tensor<float>(n, n_out);
        for(int i=0;i<n;i++){
            for(int k=0;k<n_in;k++){
                for(int j=0;j<n_out;j++){
                    output.at(i, j) += input.at(i, k) * W.at(k, j);
                }
            }
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<n_out;j++){
                output.at(i, j) += b.at(j);
            }
        }
    }

    virtual void load_weights(std::istringstream& ss){
        assert(!ss.eof());
        for(int i=0;i<n_in;i++){
            for(int j=0;j<n_out;j++){
                ss >> W.at(i, j);
            }
        }
        for(int j=0;j<n_out;j++){
            ss >> b.at(j);
        }
    }

private:
    Tensor<float> W, b;
    int n_in, n_out;
};

class Convolution2D: public layer{
public:
    Convolution2D(int nb_filter, int nb_row, int nb_col, std::vector<int> shape=std::vector<int>()){
        output_shape.resize(3);
        output_shape[0] = nb_filter;
        n_row = nb_row;
        n_col = nb_col;
        if(shape.size() == 3){
            input_shape = shape;
        }
    }

    virtual std::vector<int> get_output_shape() {
        assert(input_shape.size() == 3);
        output_shape[1] = input_shape[1] - n_row/2*2;
        output_shape[2] = input_shape[2] - n_col/2*2;
        assert(output_shape[1] > 0 && output_shape[2] > 0);
        return output_shape;
    }

    virtual void build(){
        assert(input_shape.size() == 3);
        std::vector<int> filter_shape(4);
        filter_shape[0] = output_shape[0];
        filter_shape[1] = input_shape[0];
        filter_shape[2] = n_row;
        filter_shape[3] = n_col;
        filters = Tensor<float>(filter_shape);
        biases = Tensor<float>(output_shape[0]);
    }

    virtual void forward(Tensor<float>& input){
        std::vector<int> batch_shape = output_shape;
        batch_shape.insert(batch_shape.begin(), input.shape[0]);
        output = Tensor<float>(batch_shape);
        std::vector<int> input_idx(4), filter_idx(4);
        int opos = 0;
        for(int i=0;i<output.shape[0];i++){
            input_idx[0] = i;
            for(int fo=0;fo<output.shape[1];fo++){
                filter_idx[0] = fo;
                for(int ro=0;ro<output.shape[2];ro++){
                    for(int co=0;co<output.shape[3];co++){
                        float sum = 0;
                        for(int fi=0;fi<input.shape[1];fi++){
                            input_idx[1] = input_idx[0] * input.shape[1] + fi;
                            filter_idx[1] = filter_idx[0] * filters.shape[1] + fi;
                            for(int rk=0;rk<n_row;rk++){
                                assert(ro + rk < input.shape[2]);
                                input_idx[2] = input_idx[1] * input.shape[2] + ro + n_row - 1 - rk;
                                input_idx[3] = input_idx[2] * input.shape[3] + co + n_col - 1;
                                filter_idx[2] = filter_idx[1] * n_row + rk;
                                filter_idx[3] = filter_idx[2] * n_col;
                                for(int ck=0;ck<n_col;ck++){
                                    assert(co + ck < input.shape[3]);
                                    sum += filters.ix(filter_idx[3]++) * input.ix(input_idx[3]--);
                                }
                            }
                        }
                        output.ix(opos++) = sum + biases.at(fo);
                    }
                }
            }
        }
        assert(opos == output.n);
    }

    virtual void load_weights(std::istringstream& ss){
        assert(!ss.eof());
        for(int i=0;i<filters.n;i++){
            ss >> filters.ix(i);
        }
        for(int i=0;i<biases.n;i++){
            ss >> biases.ix(i);
        }
    }

private:
    int n_row, n_col;
    Tensor<float> filters;
    Tensor<float> biases;
};


class MaxPooling2D: public layer{
public:
    MaxPooling2D(int pool_size_y, int pool_size_x){
        pool_size.resize(2);
        pool_size[0] = pool_size_y;
        pool_size[1] = pool_size_x;
    }

    virtual std::vector<int> get_output_shape() {
        assert(input_shape.size() == 3);
        output_shape = input_shape;
        // floor operation
        output_shape[1] /= pool_size[0];
        output_shape[2] /= pool_size[1];
        assert(output_shape[1] > 0 && output_shape[2] > 0);
        return output_shape;
    }

    virtual void forward(Tensor<float>& input){
        std::vector<int> batch_shape = output_shape;
        batch_shape.insert(batch_shape.begin(), input.shape[0]);
        output = Tensor<float>(batch_shape);
        for(int i=0;i<output.n;i++){
            output.ix(i) = -10000;
        }
        int max_y = output.shape[2] * pool_size[0];
        int max_x = output.shape[3] * pool_size[1];
        int opos = 0, ipos = 0;
        for(int i=0;i<input.shape[0];i++){
            for(int c=0;c<input.shape[1];c++){
                for(int y=0;y<max_y;y++){
                    for(int x=0;x<max_x;x++){
                        output.ix(opos) = std::max(output.ix(opos), input.ix(ipos));
                        ipos++;
                        if((x+1)%pool_size[1] == 0) opos++;
                    }
                    ipos += input.shape[3] - max_x;
                    if((y+1)%pool_size[0] != 0) opos -= output.shape[3];
                }
                ipos += (input.shape[2] - max_y) * input.shape[3];
            }
            assert((i+1)*input.shape[1]*input.shape[2]*input.shape[3] == ipos);
        }
    }

private:
    std::vector<int> pool_size;
};


class Flatten: public layer{
public:
    Flatten(){}

    virtual std::vector<int> get_output_shape() {
        output_shape.resize(1);
        output_shape[0] = 1;
        for(int i=0;i<input_shape.size();i++){
            output_shape[0] *= input_shape[i];
        }
        return output_shape;
    }

    virtual void forward(Tensor<float>& input){
        output = input;
        std::vector<int> batch_shape(2);
        batch_shape[0] = input.shape[0];
        batch_shape[1] = output_shape[0];
        output.reshape(batch_shape);
    }
};


class Dropout: public layer{
public:
    Dropout(float r) {
        drop_rate = r;
    }

    // a Dropout layer returns the same tersor as shape
    virtual std::vector<int> get_output_shape() {
        return input_shape;
    }

    virtual void forward(Tensor<float>& input){
        // Keras doesn't use drop_rate in test phase.
        output = input;
        // output = Tensor<float>(input.shape);
        // for(int i=0;i<input.n;i++){
        //     output.ix(i) = drop_rate * input.ix(i);
        // }
    }

private:
    float drop_rate;
};


class Activation: public layer{
    // an activation layer returns the same tersor as shape
    virtual std::vector<int> get_output_shape() {
        return input_shape;
    }
};


class Relu: public Activation{
public:
    Relu() {}
    virtual void forward(Tensor<float>& input){
        output = Tensor<float>(input.shape);
        for(int i=0;i<input.n;i++){
            output.ix(i) = std::max(0.0f, input.ix(i));
        }
    }
};


class Softmax: public Activation{
public:
    Softmax() {}
    virtual void forward(Tensor<float>& input){
        assert(input.shape.size() == 2);
        output = Tensor<float>(input.shape);
        for(int i=0;i<input.shape[0];i++){
            float max_v = input.at(i, 0);
            for(int j=1;j<input.shape[1];j++){
                max_v = std::max(max_v, input.at(i, j));
            }
            float sum_v = 0;
            for(int j=0;j<input.shape[1];j++){
                output.at(i, j) = std::exp(input.at(i, j) - max_v);
                sum_v += output.at(i, j);
            }
            for(int j=0;j<input.shape[1];j++){
                output.at(i, j) /= sum_v;
            }
        }
    }
};


class Network{
public:
    std::vector<layer*> layers;
    bool load_completed;

    Network() {
      load_completed = false;
    }

    ~Network() {
        for(int i=0;i<layers.size();i++){
            delete layers[i];
        }
    }

    void build(){
        if(layers.size() == 0) return;
        // set input shape
        for(int i=1;i<layers.size();i++){
            layers[i]->set_input_shape(layers[i-1]->get_output_shape());
        }
        // build
        for(int i=0;i<layers.size();i++){
            layers[i]->build();
        }
    }

    void load_weights(std::string filename){
        std::ifstream ifs(filename.c_str(), std::ifstream::in);
        std::string line;
        std::getline(ifs, line);
        std::istringstream ss(line);
        for(int i=0;i<layers.size();i++){
            layers[i]->load_weights(ss);
        }
	load_completed = true;
    }

    Tensor<float> predict(Tensor<float>& X){
        layers[0]->forward(X);
        for(int i=1;i<layers.size();i++){
            layers[i]->forward(layers[i-1]->output);
        }
        return layers[layers.size()-1]->output;
    }

    std::vector<int> predict_classes(Tensor<float>& X){
        int n = X.shape[0];
        Tensor<float> pred = predict(X);
        std::vector<int> label(n);
        for(int i=0;i<n;i++){
            int max_idx = 0;
            float max_value = pred.at(i, 0);
            for(int j=1;j<pred.shape[1];j++){
                if (max_value < pred.at(i, j)){
                    max_idx = j;
                    max_value = pred.at(i, j);
                }
            }
            label[i] = max_idx;
        }
        return label;
    }

    void add(layer *l){
        layers.push_back(l);
    }
};

#endif /* FORWARD_CNN_H */
