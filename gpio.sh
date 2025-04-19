#!/bin/bash

while true
do
  gpiomon -r -n 1 gpiochip0 105 | while read line; do echo "event $line"; done
done