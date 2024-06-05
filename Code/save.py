import os


def save_result(args, results, final_results):
    # create directory
    result_savepath = args.savepath
    if not os.path.exists(result_savepath):
        os.makedirs(result_savepath)
    file = result_savepath + '/results.txt'
    # according param save file
    with open(file, 'a') as f:
        f.write("lr: {}, lr_cls: {}, tau: {}, m: {}, k: {}\n".
                format(args.lr, args.lr_cls, args.tau, args.m, args.k))
        for fold, evaluation in results.items():
            f.write(
                '{}:\tAupr:{:.5f}\tAUC:{:.5f}\tF1_Score:{:.5f}\tACC:{:.5f}\tRecall:{:.5f}\t'
                'Specificity:{:.5f}\tPrecision:{:.5f}\n'.format(
                    fold, evaluation["Aupr"], evaluation["AUC"], evaluation["F1_Score"], evaluation["ACC"],
                    evaluation["Recall"], evaluation["Specificity"], evaluation["Precision"]))

        f.write('Final result:\tAupr:{:.5f}\tAUC:{:.5f}\tF1_Score:{:.5f}\tACC:{:.5f}\tRecall:{:.5f}\t'
                'Specificity:{:.5f}\tPrecision:{:.5f}\n'.format(final_results["Aupr"], final_results["AUC"],
                                                                final_results["F1_Score"], final_results["ACC"],
                                                                final_results["Recall"],
                                                                final_results["Specificity"],
                                                                final_results["Precision"]))
