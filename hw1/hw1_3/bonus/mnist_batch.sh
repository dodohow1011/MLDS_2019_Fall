for i in 10 20 30 60 80 100 200 300 600 800 1000;

do 
	python3 mnist.py -b $i
done

python3 mnist_train_loss.py
python3 sharpness_plot.py