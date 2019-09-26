import per_action2
import importlib
import dl
import pickle
import traceback, sys
import rospy
from std_msgs.msg import Float32

if 1: #__name__ == "__main__":
# if __name__ == "__main__":
    try:

        myoptions = []
        thisop = per_action2.Options()
        thisop.settrain('hmdb14')
        thisop.settest('hmdb14')
        thisop.modeltype = '0-hiddenD'
        # thisop.modeltype = '0-hidden2' ## with the ReLU
        thisop.mytester = per_action2.getcf
        thisop.learning_rate = 0.0001
        thisop.num_epochs = 500
        thisop.num_batches = 100

        # thisop.mytester = per_action2.getcf2
        thisop.choose_list = ['brush_hair','chew','clap','drink','eat','jump','pick','pour','sit','smile','stand','talk','walk','wave']
        thisop.evaluation_mode = "both"
        myoptions.append(thisop)
        #
        # thisop = Options()
        # thisop.settrain('hmdb14')
        # thisop.settest('myset')
        # myoptions.append(thisop)
        #
        # thisop = Options()
        # thisop.settrain('both')
        # thisop.settest('myset')
        # myoptions.append(thisop)
        #
        # thisop = Options()
        # thisop.settrain('myset')
        # thisop.settest('myset')
        # myoptions.append(thisop)
        results = []
        out = []
    # hmdb51testscores_own_flow_split_1_y_yhat1559680138.96.obj
    # hmdb51testscores_own_flow_split_2_y_yhat1559681846.25.obj
    # hmdb51testscores_own_flow_split_3_y_yhat1559683493.16.obj
    # hmdb51testscores_own_rgb_split_1_y_yhat1559680057.25.obj
    # hmdb51testscores_own_rgb_split_2_y_yhat1559681761.51.obj
    # hmdb51testscores_own_rgb_split_3_y_yhat1559683414.2.obj
    # hmdb51trainscores_own_flow_split_1_y_yhat1558476156.41.obj
    # hmdb51trainscores_own_flow_split_2_y_yhat1558489246.06.obj
    # hmdb51trainscores_own_flow_split_3_y_yhat1558502294.88.obj
    # hmdb51trainscores_own_rgb_split_1_y_yhat1558475489.24.obj
    # hmdb51trainscores_own_rgb_split_2_y_yhat1558488594.63.obj
    # hmdb51trainscores_own_rgb_split_3_y_yhat1558501624.87.obj


        for z,thisOptions in enumerate(myoptions):
            #options_pub.publish(z/float(len(myoptions)))
            thisOptions.NUM = z
            for i in range(1,4):
                #splitpub.publish(i)
                thisOptions.split = i
                if i == 1:    ## split 1 train
                    ## hmdb51, myset/ actually hmdb51 is hmdb14!
                    #hmdb train
                    #myset train
                    #hmdb test
                    #myset test
                    a,thisOptions,thisOuta = thisOptions.mytester([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_rgb_split_1_y_yhat1558475489.24.obj',
                    '/mnt/share/ar/datasets/df_per_action/it shouldnt try to load this!'],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_rgb_split_1_y_yhat1559680057.25.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_rgbsplit_1_y_yhat1557369944.32.obj'], thisOptions)

                    b,thisOptions,thisOutb = thisOptions.mytester([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_flow_split_1_y_yhat1558476156.41.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_train_flowsplit_1_y_yhat1557353106.0.obj',],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_flow_split_1_y_yhat1559680138.96.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_flowsplit_1_y_yhat1557369955.06.obj'], thisOptions)
                    totala = a
                    totalb = b

                if i == 2:# ## split 2 train
                    a,thisOptions,thisOuta = thisOptions.mytester([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_rgb_split_2_y_yhat1558488594.63.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_train_rgbsplit_2_y_yhat1557353417.19.obj'],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_rgb_split_2_y_yhat1559681761.51.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_rgbsplit_2_y_yhat1557370097.39.obj'], thisOptions)

                    b,thisOptions,thisOutb = thisOptions.mytester([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_flow_split_2_y_yhat1558489246.06.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_train_flowsplit_2_y_yhat1557353433.46.obj'],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_flow_split_2_y_yhat1559681846.25.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_flowsplit_2_y_yhat1557370107.53.obj'], thisOptions)
                    totala = a+ totala
                    totalb = b+ totalb

                if i == 3:# ## split 3 train
                    a,thisOptions,thisOuta = thisOptions.mytester([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_rgb_split_3_y_yhat1558501624.87.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_train_rgbsplit_3_y_yhat1557362421.94.obj'],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_rgb_split_3_y_yhat1559683414.2.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_rgbsplit_3_y_yhat1557370257.33.obj'], thisOptions)

                    b,thisOptions,thisOutb = thisOptions.mytester([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_flow_split_3_y_yhat1558502294.88.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_train_flowsplit_3_y_yhat1557362437.09.obj'],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_flow_split_3_y_yhat1559683493.16.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_flowsplit_3_y_yhat1557370264.5.obj'], thisOptions)
                    totala = a+ totala
                    totalb = b+ totalb
                print('############')
                results.append(a)
                results.append(b)
                out.append(thisOuta)
                out.append(thisOutb)

            print('############')
            results.append(totala)
            results.append(totalb)
            ####tests only one
            #break

        dl.tic()
        with open('results_3splits%s.obj'%dl.now,'wb') as f:
            pickle.dump(results,f)
            pickle.dump(myoptions,f)
            pickle.dump(out,f)

    # except BaseException as E:
    #     print(E)
    except:
        traceback.print_exc(file=sys.stdout)
