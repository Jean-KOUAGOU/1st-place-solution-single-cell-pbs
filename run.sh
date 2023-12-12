#!/bin/bash
sudo docker run --shm-size 8G --name single_cell_container single_cell_env $1
sudo docker cp single_cell_container:/app/submissions/submission.csv ./submission.csv