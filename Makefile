# A Makefile because I'm too lazy to type

HOSTNAME=$(shell cat /proc/sys/kernel/hostname)

clean:
	@rm ./logs/*
	@rm ./models/*

tsb:
	@tensorboard --logdir=./logs
	@xdg-open http://$(HOSTNAME):6006
