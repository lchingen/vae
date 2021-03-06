# A Makefile because I'm too lazy to type

HOSTNAME=$(shell cat /proc/sys/kernel/hostname)

clean:
	@rm -rf ./logs/*

tsb:
	@tensorboard --logdir=./logs

build:
	@python3 build.py

train:
	@python3 train.py

test:
	@python3 test.py

gen:
	@python3 generate.py

custom:
	@python3 custom.py
