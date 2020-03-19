#!/bin/bash
set -ex
git checkout origin/develop
git pull origin develop
git remote prune origin
