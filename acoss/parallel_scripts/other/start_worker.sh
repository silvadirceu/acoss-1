#!/bin/bash

ray start --redis-address=$1

echo "Worker ${2} PID: $$"
echo "$$" | tee ~/pid_storage/worker${2}.pid
sleep infinity

echo "Worker ${2} stopped"