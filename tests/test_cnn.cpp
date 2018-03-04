#include <vector>
#include <cmath>
#include "../src/forward_cnn.h"

int read_int(std::ifstream& ifs){
    int n;
    ifs.read(reinterpret_cast<char*>(&n), sizeof(int));
    return n;
}

void load_tensor(Tensor<float>& data, std::ifstream& ifs){
    std::string line;
    std::getline(ifs, line);
    std::istringstream ss(line);
    for(int i=0;i<data.n;i++){
        ss >> data.ix(i);
    }
}

int main(){

    std::cout << "Build and load weights" << std::endl;
    std::ifstream ifs("data/cnn_weights.bin");
    int nb_classes = read_int(ifs);

    std::vector<std::string> unique_labels(nb_classes);
    for(int i=0;i<nb_classes;i++){
        int str_len = read_int(ifs);
        std::vector<char> str(str_len);
        ifs.read(str.data(), sizeof(char) * str_len);
        unique_labels[i].assign(str.begin(), str.end());
    }

    Network *net;
    net = new Network();

    std::vector<int> input_shape(3);
    input_shape[0] = 1;
    input_shape[1] = 48;
    input_shape[2] = 48;

    net->add(new Convolution2D(32, 9, 9, input_shape));
    net->add(new Relu());
    net->add(new MaxPooling2D(2, 2));
    net->add(new Dropout(0.5));

    net->add(new Convolution2D(64, 5, 5));
    net->add(new Relu());
    net->add(new MaxPooling2D(2, 2));
    net->add(new Dropout(0.5));

    net->add(new Convolution2D(128, 3, 3));
    net->add(new Relu());
    net->add(new MaxPooling2D(2, 2));
    net->add(new Dropout(0.5));

    net->add(new Flatten());
    net->add(new Dense(128));
    net->add(new Relu());
    net->add(new Dropout(0.5));

    net->add(new Dense(nb_classes));
    net->add(new Softmax());

    net->build();
    net->load_weights(ifs);
    net->set_label(unique_labels);

    std::cout << "Load input" << std::endl;

    std::ifstream ifs_for_ss("data/cnn_data.txt");

    std::vector<int> X_input_shape(4);
    X_input_shape[0] = 2;
    X_input_shape[1] = 1;
    X_input_shape[2] = 48;
    X_input_shape[3] = 48;
    Tensor<float> X(X_input_shape);
    load_tensor(X, ifs_for_ss);

    std::vector<int> layer_idx;
    layer_idx.push_back(1);
    layer_idx.push_back(2);
    layer_idx.push_back(3);
    layer_idx.push_back(5);
    layer_idx.push_back(6);
    layer_idx.push_back(7);
    layer_idx.push_back(9);
    layer_idx.push_back(10);
    layer_idx.push_back(11);
    layer_idx.push_back(12);
    layer_idx.push_back(14);
    layer_idx.push_back(15);
    layer_idx.push_back(17);

    std::cout << "Forward calculation" << std::endl;
    net->predict(X);

    std::cout << "Check hidden outputs" << std::endl;
    for(int k=0;k<layer_idx.size();k++){
        int i = layer_idx[k];
        Tensor<float> data(net->layers[i]->output.shape);
        load_tensor(data, ifs_for_ss);
        for(int j=0;j<data.n;j++){
            float diff = std::abs(data.ix(j) - net->layers[i]->output.ix(j));
            if(diff > 1e-5){
                std::cout << "Fail test" << std::endl;
                printf("layer: %d, ix: %d, python output: %.8f, c++ output: %.8f\n",
                       i, j, data.ix(j), net->layers[i]->output.ix(j));
                exit(1);
            }
        }
    }

    std::cout << "Pass test!" << std::endl;
    return 0;
}
