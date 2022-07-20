# Area 1
python main_S3DIS.py --mode train --gpu 4 --test_area 1 --labeled_point 0.1%
python main_S3DIS.py --mode test --gpu 4 --test_area 1

python main_S3DIS.py --mode test --gpu 4 --test_area 2 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 3 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 4 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 5 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 6 --gen_pesudo
python main_S3DIS.py --mode train --gpu 4 --test_area 1 --labeled_point 0.1% --retrain
python main_S3DIS.py --mode test --gpu 4 --test_area 1 --labeled_point 0.1%

# Area 2
python main_S3DIS.py --mode train --gpu 4 --test_area 2 --labeled_point 0.1%
python main_S3DIS.py --mode test --gpu 4 --test_area 2

python main_S3DIS.py --mode test --gpu 4 --test_area 1 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 3 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 4 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 5 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 6 --gen_pesudo
python main_S3DIS.py --mode train --gpu 4 --test_area 2 --labeled_point 0.1% --retrain
python main_S3DIS.py --mode test --gpu 4 --test_area 2 --labeled_point 0.1%

# Area 3
python main_S3DIS.py --mode train --gpu 4 --test_area 3 --labeled_point 0.1%
python main_S3DIS.py --mode test --gpu 4 --test_area 3

python main_S3DIS.py --mode test --gpu 4 --test_area 1 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 2 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 4 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 5 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 6 --gen_pesudo
python main_S3DIS.py --mode train --gpu 4 --test_area 3 --labeled_point 0.1% --retrain
python main_S3DIS.py --mode test --gpu 4 --test_area 3 --labeled_point 0.1%

# Area 4
python main_S3DIS.py --mode train --gpu 4 --test_area 4 --labeled_point 0.1%
python main_S3DIS.py --mode test --gpu 4 --test_area 4

python main_S3DIS.py --mode test --gpu 4 --test_area 1 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 2 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 3 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 5 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 6 --gen_pesudo
python main_S3DIS.py --mode train --gpu 4 --test_area 4 --labeled_point 0.1% --retrain
python main_S3DIS.py --mode test --gpu 4 --test_area 4 --labeled_point 0.1%

# Area 6
python main_S3DIS.py --mode train --gpu 4 --test_area 6 --labeled_point 0.1%
python main_S3DIS.py --mode test --gpu 4 --test_area 6

python main_S3DIS.py --mode test --gpu 4 --test_area 1 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 2 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 3 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 4 --gen_pesudo
python main_S3DIS.py --mode test --gpu 4 --test_area 5 --gen_pesudo
python main_S3DIS.py --mode train --gpu 4 --test_area 6 --labeled_point 0.1% --retrain
python main_S3DIS.py --mode test --gpu 4 --test_area 6 --labeled_point 0.1%