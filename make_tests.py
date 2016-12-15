import numpy as np
import pprint

from keras.models import Model
from keras.models import Sequential
from keras.layers import merge, Input
from keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D
from keras.layers.advanced_activations import ELU

from kerasify import export_model

np.set_printoptions(precision=25, threshold=np.nan)

def c_array(a):
    s = pprint.pformat(a.flatten())
    s = s.replace('[', '{').replace(']', '}').replace('array(', '').replace(')', '').replace(', dtype=float32', '')

    shape = ''

    if a.shape == ():
        s = '{%s}' % s
        shape = '(1)'
    else:
        shape = repr(a.shape).replace(',)', ')')

    return shape, s


TEST_CASE = '''
bool test_%s(double* load_time, double* apply_time)
{
    printf("TEST %s\\n");

    KASSERT(load_time, "Invalid double");
    KASSERT(apply_time, "Invalid double");

    Tensor in%s;
    in.data_ = %s;

    Tensor out%s;
    out.data_ = %s;

    KerasTimer load_timer;
    load_timer.Start();

    KerasModel model;
    KASSERT(model.LoadModel("test_%s.model"), "Failed to load model");

    *load_time = load_timer.Stop();

    KerasTimer apply_timer;
    apply_timer.Start();

    Tensor predict = out;
    KASSERT(model.Apply(&in, &out), "Failed to apply");

    *apply_time = apply_timer.Stop();

    for (int i = 0; i < out.dims_[0]; i++)
    {
        KASSERT_EQ(out(i), predict(i), %s);
    }

    return true;
}
'''

def output_testcase(model, test_x, test_y, name, eps):
    print "Processing %s" % name
    model.compile(loss='mean_squared_error', optimizer='adamax')
    model.fit(test_x, test_y, nb_epoch=1, verbose=False)
    predict_y = model.predict(test_x).astype('f')
    print model.summary()

    export_model(model, 'test_%s.model' % name)

    with open('test_%s.h' % name, 'w') as f:
        x_shape, x_data = c_array(test_x[0])
        y_shape, y_data = c_array(predict_y[0])

        f.write(TEST_CASE % (name, name, x_shape, x_data, y_shape, y_data, name, eps))


TEST_CASE_EX = '''
bool test_%s(double* load_time, double* apply_time)
{
    printf("TEST %s\\n");

    KASSERT(load_time, "Invalid double");
    KASSERT(apply_time, "Invalid double");

    std::vector<Tensor> in_tensors = {%s};
    std::vector<Tensor> expected = {%s};

    KerasTimer load_timer;
    load_timer.Start();

    KerasModel model;
    KASSERT(model.LoadModel("test_%s.model"), "Failed to load model");

    *load_time = load_timer.Stop();

    KerasTimer apply_timer;
    apply_timer.Start();

    std::vector<Tensor> predicted = expected;
    TensorMap in;
    for (unsigned int i = 0; i < in_tensors.size(); i++)
    {
        in["in" + std::to_string(i)] = &(in_tensors[i]);
    }
    TensorMap out;
    for (unsigned int i = 0; i < predicted.size(); i++)
    {
        out["out" + std::to_string(i)] = &(predicted[i]);
    }
    KASSERT(model.Apply(in, &out), "Failed to apply");

    *apply_time = apply_timer.Stop();

    for (unsigned int i = 0; i < expected.size(); i++)
    {
        Tensor& expect = expected[i];
        Tensor& predict = predicted[i];
        for (int j = 0; j < expect.dims_[0]; j++)
        {
            KASSERT_EQ(expect(j), predict(j), %s);
        }
    }

    return true;
}
'''

def c_array_init(a):
    shape, s = c_array(a)
    shape = shape.replace('(', '{').replace(')', '}')
    return shape, s

def output_testcase_ex(model, test_x_list, test_y_list, name, eps):
    print "Processing %s" % name
    model.compile(loss='mean_squared_error', optimizer='adamax')
    model.fit(test_x_list, test_y_list, nb_epoch=1, verbose=False)
    predict_y_list = model.predict(test_x_list)
    print model.summary()

    export_model(model, 'test_%s.model' % name)

    with open('test_%s.h' % name, 'w') as f:
        x = ["{%s, %s}" % c_array_init(test_x) for test_x in test_x_list]
        x = ','.join(x)
        y = ["{%s, %s}" % c_array_init(predict_y) for predict_y in predict_y_list]
        y = ','.join(y)

        f.write(TEST_CASE_EX % (name, name, x, y, name, eps))


# ''' Dense 1x1 '''
# test_x = np.arange(10)
# test_y = test_x * 10 + 1
# model = Sequential()
# model.add(Dense(1, input_dim=1))

# output_testcase(model, test_x, test_y, 'dense_1x1', '1e-6')

# ''' Functional Dense 1x1 '''
# test_x = np.arange(10)
# test_y = test_x * 10 + 1

# input = Input(name='in1', shape=(1,))
# output = Dense(1, name='out1')(input)
# model = Model(input=input, output=output)

# output_testcase(model, test_x, test_y, 'functional_dense_1x1', '1e-6')

# ''' Functional Dense 1x1 Merge'''
# test_x = np.arange(10)
# test_y = test_x * 10 + 1

# input = Input(name='in1', shape=(1,))
# h1 = Dense(1, name='hidden1')(input)
# h2 = Dense(1, name='hidden2')(h1)
# hA = Dense(1, name='hiddenA')(input)
# m =  merge([h2, hA], mode='concat', concat_axis=-1)
# output = Dense(1, name='out1')(m)
# model = Model(input=input, output=output)

# output_testcase(model, test_x, test_y, 'functional_dense_1x1_merge', '1e-6')

''' Functional Dense 1x1 Extended'''
test_x_list = [np.arange(10), np.arange(10)]
test_y_list = [test_x_list[0] * 10 + 1, test_x_list[1] * 5 + 2]

input0 = Input(name='in0', shape=(1,))
input1 = Input(name='in1', shape=(1,))
h1 = Dense(1, name='hidden1')(input0)
h2 = Dense(1, name='hidden2')(h1)

hA = Dense(1, name='hiddenA')(input1)
m =  merge([h2, hA], mode='concat', concat_axis=-1)
output0 = Dense(1, name='out0')(m)
output1 = Dense(1, name='out1')(m)
model = Model(input=[input0, input1], output=[output0, output1])

output_testcase_ex(model, test_x_list, test_y_list, 'functional_dense_1x1_merge_extended', '1e-6')

# ''' Dense 10x1 '''
# test_x = np.random.rand(10, 10).astype('f')
# test_y = np.random.rand(10).astype('f')
# model = Sequential()
# model.add(Dense(1, input_dim=10))

# output_testcase(model, test_x, test_y, 'dense_10x1', '1e-6')

# ''' Dense 2x2 '''
# test_x = np.random.rand(10, 2).astype('f')
# test_y = np.random.rand(10).astype('f')
# model = Sequential()
# model.add(Dense(2, input_dim=2))
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'dense_2x2', '1e-6')

# ''' Dense 10x10 '''
# test_x = np.random.rand(10, 10).astype('f')
# test_y = np.random.rand(10).astype('f')
# model = Sequential()
# model.add(Dense(10, input_dim=10))
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'dense_10x10', '1e-6')

# ''' Dense 10x10x10 '''
# test_x = np.random.rand(10, 10).astype('f')
# test_y = np.random.rand(10, 10).astype('f')
# model = Sequential()
# model.add(Dense(10, input_dim=10))
# model.add(Dense(10))

# output_testcase(model, test_x, test_y, 'dense_10x10x10', '1e-6')

# ''' Conv 2x2 '''
# test_x = np.random.rand(10, 1, 2, 2).astype('f')
# test_y = np.random.rand(10, 1).astype('f')
# model = Sequential()
# model.add(Convolution2D(1, 2, 2, input_shape=(1, 2, 2)))
# model.add(Flatten())
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'conv_2x2', '1e-6')

# ''' Conv 3x3 '''
# test_x = np.random.rand(10, 1, 3, 3).astype('f').astype('f')
# test_y = np.random.rand(10, 1).astype('f')
# model = Sequential()
# model.add(Convolution2D(1, 3, 3, input_shape=(1, 3, 3)))
# model.add(Flatten())
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'conv_3x3', '1e-6')

# ''' Conv 3x3x3 '''
# test_x = np.random.rand(10, 3, 10, 10).astype('f')
# test_y = np.random.rand(10, 1).astype('f')
# model = Sequential()
# model.add(Convolution2D(3, 3, 3, input_shape=(3, 10, 10)))
# model.add(Flatten())
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'conv_3x3x3', '1e-6')

# ''' Activation ELU '''
# test_x = np.random.rand(1, 10).astype('f')
# test_y = np.random.rand(1, 1).astype('f')
# model = Sequential()
# model.add(Dense(10, input_dim=10))
# model.add(ELU(alpha=0.5))
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'elu_10', '1e-6')

# ''' Activation relu '''
# test_x = np.random.rand(1, 10).astype('f')
# test_y = np.random.rand(1, 10).astype('f')
# model = Sequential()
# model.add(Dense(10, input_dim=10))
# model.add(Activation('relu'))

# output_testcase(model, test_x, test_y, 'relu_10', '1e-6')

# ''' Dense relu '''
# test_x = np.random.rand(1, 10).astype('f')
# test_y = np.random.rand(1, 10).astype('f')
# model = Sequential()
# model.add(Dense(10, input_dim=10, activation='relu'))
# model.add(Dense(10, input_dim=10, activation='relu'))
# model.add(Dense(10, input_dim=10, activation='relu'))

# output_testcase(model, test_x, test_y, 'dense_relu_10', '1e-6')

# ''' Conv softplus '''
# test_x = np.random.rand(10, 1, 2, 2).astype('f')
# test_y = np.random.rand(10, 1).astype('f')
# model = Sequential()
# model.add(Convolution2D(1, 2, 2, input_shape=(1, 2, 2), activation='softplus'))
# model.add(Flatten())
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'conv_softplus_2x2', '1e-6')


# ''' Maxpooling2D 1x1'''
# test_x = np.random.rand(10, 1, 10, 10).astype('f')
# test_y = np.random.rand(10, 1).astype('f')
# model = Sequential()
# model.add(MaxPooling2D(pool_size=(1, 1), input_shape=(1, 10, 10)))
# model.add(Flatten())
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'maxpool2d_1x1', '1e-6')

# ''' Maxpooling2D 2x2'''
# test_x = np.random.rand(10, 1, 10, 10).astype('f')
# test_y = np.random.rand(10, 1).astype('f')
# model = Sequential()
# model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(1, 10, 10)))
# model.add(Flatten())
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'maxpool2d_2x2', '1e-6')

# ''' Maxpooling2D 3x2x2'''
# test_x = np.random.rand(10, 3, 10, 10).astype('f')
# test_y = np.random.rand(10, 1).astype('f')
# model = Sequential()
# model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(3, 10, 10)))
# model.add(Flatten())
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'maxpool2d_3x2x2', '1e-6')

# ''' Maxpooling2D 3x3x3'''
# test_x = np.random.rand(10, 3, 10, 10).astype('f')
# test_y = np.random.rand(10, 1).astype('f')
# model = Sequential()
# model.add(MaxPooling2D(pool_size=(3, 3), input_shape=(3, 10, 10)))
# model.add(Flatten())
# model.add(Dense(1))

# output_testcase(model, test_x, test_y, 'maxpool2d_3x3x3', '1e-6')


# ''' Benchmark '''
# test_x = np.random.rand(1, 3, 128, 128).astype('f')
# test_y = np.random.rand(1, 10).astype('f')
# model = Sequential()
# model.add(Convolution2D(16, 7, 7, input_shape=(3, 128, 128), activation='relu'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(ELU())
# model.add(Convolution2D(8, 3, 3))
# model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(10))

# output_testcase(model, test_x, test_y, 'benchmark', '1e-3')


