import sys

weka_predictions_path = sys.argv[1]
output_predictions_path = sys.argv[2]

with open(weka_predictions_path, 'r') as weka_file, open(output_predictions_path, 'w') as output_file:
    for line in weka_file.readlines()[1:-1]:
        inst, actual, predicted, error, *distribution = line.split(',')
        class_probabilities = [class_probability.replace('*', '').replace('\n', '') for class_probability in distribution]
        class_index, class_name = predicted.split(':')
        output_file.write(','.join(class_probabilities + [class_name + '\n']))

"""
inst#,actual,predicted,error,distribution,,
1,1:Iris-setosa,1:Iris-setosa,,*0.958,0.021,0.021
2,1:Iris-setosa,1:Iris-setosa,,*0.958,0.021,0.021
3,1:Iris-setosa,1:Iris-setosa,,*0.958,0.021,0.021
4,1:Iris-setosa,1:Iris-setosa,,*0.958,0.021,0.021
5,1:Iris-setosa,1:Iris-setosa,,*0.958,0.021,0.021
6,2:Iris-versicolor,2:Iris-versicolor,,0.021,*0.936,0.043
7,2:Iris-versicolor,2:Iris-versicolor,,0.021,*0.936,0.043
8,2:Iris-versicolor,2:Iris-versicolor,,0.021,*0.936,0.043
9,2:Iris-versicolor,2:Iris-versicolor,,0.021,*0.936,0.043
10,2:Iris-versicolor,2:Iris-versicolor,,0.021,*0.936,0.043
11,3:Iris-virginica,3:Iris-virginica,,0.022,0.044,*0.933
12,3:Iris-virginica,3:Iris-virginica,,0.022,0.044,*0.933
13,3:Iris-virginica,2:Iris-versicolor,+,0.021,*0.936,0.043
14,3:Iris-virginica,3:Iris-virginica,,0.022,0.044,*0.933
15,3:Iris-virginica,3:Iris-virginica,,0.022,0.044,*0.933
"""