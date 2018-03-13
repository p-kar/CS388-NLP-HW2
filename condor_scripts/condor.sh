universe = vanilla
Initialdir = /scratch/cluster/pkar/CS388-NLP-HW2/
Executable = /lusr/bin/bash
Arguments = /scratch/cluster/pkar/CS388-NLP-HW2/condor_scripts/task.sh
+Group   = "GRAD"
+Project = "INSTRUCTIONAL"
+ProjectDescription = "HW2 for CS388"
Requirements = TARGET.GPUSlot
getenv = True
request_GPUs = 1
+GPUJob = true
Log = /scratch/cluster/pkar/CS388-NLP-HW2/tmp/pos_train_dir/condor.log
Error = /scratch/cluster/pkar/CS388-NLP-HW2/tmp/pos_train_dir/condor.err
Output = /scratch/cluster/pkar/CS388-NLP-HW2/tmp/pos_train_dir/condor.out
Notification = complete
Notify_user = pkar@cs.utexas.edu
Queue 1