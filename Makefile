# A Makefile because I'm too lazy to type

HOSTNAME=$(shell cat /proc/sys/kernel/hostname)

clean:
	@rm -f ./logs/*
	@rm -f ./models/*

tsb:
	@tensorboard --logdir=./logs

train:
	@make clean && python3 train.py --num_epochs 50 --batch_size 128 --dataset cifar10 && make test

test:
	@python3 test.py
