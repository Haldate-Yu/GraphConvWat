import csv
import os


def save_results(args, file_name, tst_loss, tst_rel_err, tst_rel_err_obs, tst_rel_err_hid):
    if not os.path.exists('./results/{}'.format(args.wds)):
        print("=" * 20)
        print("Creating Results File !!!")

        os.makedirs('./results/{}'.format(args.wds))

    filename = "./results/{}/{}".format(
        args.wds, file_name)

    headerList = ["Method", "Learning_Rate",
                  "Weight_Decay", "Observation_Ratio",
                  "Encoder_Layers", "Hidden_Dims",
                  "::::::::",
                  "test_loss", "test_relative_error_all",
                  "test_relative_error_obs", "test_relative_error_hid"]

    with open(filename, "a+") as f:
        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, {}, {}, {}, {}, :::::::::, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(
            args.model, args.lr, args.decay,
            args.obsrat, args.layers, args.hidden_dim,
            tst_loss, tst_rel_err,
            tst_rel_err_obs, tst_rel_err_hid
        )
        f.write(line)
