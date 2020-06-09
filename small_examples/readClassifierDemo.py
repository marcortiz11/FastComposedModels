import numpy as np
import Source.make_util as make_util
import Source.io_util as io_util
import time

# construting times:
# PROTO PERFROMANCE:
#  construct: 8.177 secs
#       show: 13.458 secs
#       save: 4.609 secs

# LOAD TIME of this script:
# PROTO PERFROMANCE:
#  load: 8.132 secs
#  show: 13.598 secs

if __name__ == '__main__':
    # Load classifier
    name = "demo_classifier"

    name = "/dataT/eid/GIT/tpml/ml_experiments/001interns/V001_DPN26_ref_0_old"
    name = "/dataT/eid/GIT/tpml/ml_experiments/001interns/V001_DPN26_ref_0"


    model = make_util.make_empty_classifier()

    print("Load ...")
    t_start = time.time()
    io_util.read_message(name, model)
    t_duration_load = time.time() - t_start

    t_start = time.time()
    print(model)
    t_duration_show = time.time() - t_start

    print("PROTO PERFROMANCE:")
    print(" load: {:.3f} secs".format(t_duration_load))
    print(" show: {:.3f} secs".format(t_duration_show))
    print("")
    print("ALL DONE")