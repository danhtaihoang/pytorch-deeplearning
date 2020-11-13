#!/bin/bash
echo -n "message: "
read message
git add --all
git commit -am "${message}"
git push origin master