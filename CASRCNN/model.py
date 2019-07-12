from utils import (
  read_data, 
  input_setup, 
  input_setup_test,
  imsave,
  preprocess,
  merge,
  getXtest
)
import glob
import numpy as np
import gc
from functools import reduce
import math
import time
import os
import tensorflow as tf
from dataLoader import dataLoader


class SRCNN(object):
    """6-1 init SRCNN and setup hyperparameters"""
    def __init__(self, 
               sess, 
               config):

        self.sess = sess
        self.config=config
        self.build_model()
        
    """6-2 define model"""
    @staticmethod
    def conv_layer(input,filter_shape,seed=111,name='1'):#filter_shape: [h,w,c_in,c_out]
        with tf.name_scope('conv'+name):
            w=tf.get_variable('W'+name,filter_shape,initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            b=tf.Variable(tf.constant(0.1,shape=[filter_shape[4]]), name='b'+name)
            conv=tf.nn.conv3d(input, w, strides=[1,1,1,1,1], padding='SAME')
            act=tf.nn.relu(conv+b)
            tf.summary.histogram('weights',w)
            tf.summary.histogram('biases',b)
            tf.summary.histogram('convs',conv)
            return act
    @staticmethod
    def conv_layer_noact(input,filter_shape,seed=111,name='1'):#filter_shape: [h,w,c_in,c_out]
        with tf.name_scope('conv'+name):
            w=tf.get_variable('W'+name,filter_shape,initializer=tf.contrib.layers.xavier_initializer(seed=seed))
            b=tf.Variable(tf.constant(0.1,shape=[filter_shape[4]]), name='b'+name)
            conv=tf.nn.conv2d(input, w, strides=[1,1,1,1,1], padding='SAME')
            tf.summary.histogram('weights',w)
            tf.summary.histogram('biases',b)
            tf.summary.histogram('convs',conv)
            tf.summary.image('output',conv)
            return conv+b
    def build_model(self):
        #input
        self.images = tf.placeholder(tf.float32, [None, self.config.image_size, self.config.image_size, self.config.image_size, 3], name='input')
        
        #tf.summary.image('input',self.images[:,:,:,(self.config.c_dim-1)//2:(self.config.c_dim-1)//2+1])
        #output
        self.labels = tf.placeholder(tf.float32, [None, self.config.label_size, self.config.label_size, 3], name='labels')
        tf.summary.image('target',self.labels)
        #weights
#        self.weights = {
#          'w1': tf.get_variable('W1',[9, 9, self.config.c_dim, 64], initializer=tf.contrib.layers.xavier_initializer(seed=111)),
#          'w2': tf.get_variable('W2',[5,5, 64, 32], initializer=tf.contrib.layers.xavier_initializer(seed=222)),
#          'w3': tf.get_variable('W3',[9, 9, 32, 1], initializer=tf.contrib.layers.xavier_initializer(seed=333))
#          }
#        #bias
#        self.biases = {
#          'b1': tf.Variable(tf.constant(0.1,shape=[64]), name='b1'),
#          'b2': tf.Variable(tf.constant(0.1,shape=[32]), name='b2'),
#          'b3': tf.Variable(tf.constant(0.1,shape=[1]), name='b3'),
#          }
        #layers
#        with tf.name_scope('conv1'):
#            self.conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
#        with tf.name_scope('conv2'):
#            self.conv2 = tf.nn.relu(tf.nn.conv2d(self.conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
#        with tf.name_scope('conv3'):
#            self.pred  = tf.nn.conv2d(self.conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3']
        self.conv1=self.conv_layer(self.images,[9,9,9,self.config.c_dim,64],seed=111,name='1')
        self.conv2=self.conv_layer(self.conv1,[1,1,1,64,32],seed=222,name='2')
        #prediction
        self.pred = self.conv_layer_noact(self.conv2,[5,5,5,32,3],seed=333,name='3')
        # Loss function (MSE) #avg per sample
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        # Stochastic gradient descent with the standard backpropagation
        with tf.name_scope('optimization'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        #to save best model
        self.saver = tf.train.Saver()

    """7-1 train/test"""
    def input_parser(self,img_path):
        img,lbl=preprocess(img_path)
        img=np.asarray([img]*self.config.c_dim).astype(np.float32)
        img=np.transpose(img,(1,2,0))#channel at tail
        return img,lbl
    
    def test_whole_img(self):
        print('whole image based testing')
        try:
            self.load(self.config.checkpoint_dir)
            print(" [*] Load SUCCESS")
        except:
            print(" [!] Load failed...")
            return
        print('new_data_folder',self.config.new_image_path)
        X_test,sameSize,namelist=getXtest(self.config.new_image_path,self.config.c_dim)
        if not sameSize:
            self.config.test_batch_size=1
        tst_data_loader=dataLoader(dataSize=len(X_test),
                                   batchSize=self.config.test_batch_size,
                                   shuffle=False)
        tst_batch_count=int(math.ceil(len(X_test)/self.config.test_batch_size))
        
        result=list()
        #self.sess.run(new_init_op)
        start_time=time.time()
        for batch in range(tst_batch_count):
            inx=tst_data_loader.get_batch()
            X=list()
            for i in np.nditer(inx):
                X.append(X_test[i])
            #X=X_test[inx]#self.sess.run(next_batch)
            y_pred = self.pred.eval({self.images: np.asarray(X)})
            result.append(y_pred)

        print("time: [%4.2f]" % (time.time()-start_time))
        #flatten list
        print(len(result))
        if self.config.test_batch_size!=1:
            output=list()
            for i in result:
                for j in range(i.shape[0]):
                    output.append(i[j])
            print(len(output))
            print(output[0].shape)
        else:
            output=result[:]
        #flatten output
        output=map(np.squeeze,output)
        #save result
        for i in range(len(output)):
            imsave(np.clip(output[i],0,1),namelist[i].replace('.bmp','.bmp.c'+str(self.config.c_dim)))
        return
        
        

    def test(self):
        print('patched based testing')
                #load new images in a folder
        try:
            self.load(self.config.checkpoint_dir)
            print(" [*] Load SUCCESS")
        except:
            print(" [!] Load failed...")
            return

        print('new_data_folder',self.config.new_image_path)

        nxny_list,namelist=input_setup_test(self.sess,self.config)
        new_data_dir = os.path.join(self.config.checkpoint_dir,'new.c'+str(self.config.c_dim)+'.h5')
        X_test,_=read_data(new_data_dir)
        tst_data_loader=dataLoader(dataSize=X_test.shape[0],
                                   batchSize=self.config.test_batch_size,
                                   shuffle=False)
        tst_batch_count=int(math.ceil(X_test.shape[0]/self.config.test_batch_size))
        #print(X_test[0].shape)
        #print(X_test[1].shape)
        #new_data_loader=tf.data.Dataset.from_tensor_slices(X_test)
        #new_data_loader = new_data_loader.batch(batch_size=self.config.test_batch_size)
        #iterator = tf.data.Iterator.from_structure(new_data_loader.output_types,new_data_loader.output_shapes)
        #next_batch=iterator.get_next()
        #new_init_op = iterator.make_initializer(new_data_loader)
        
        result=list()
        #self.sess.run(new_init_op)
        start_time=time.time()
        for batch in range(tst_batch_count):
            inx=tst_data_loader.get_batch()
            X=X_test[inx].view()#self.sess.run(next_batch)
            y_pred = self.pred.eval({self.images: X})
            result.append(y_pred)
                #total_mse+=tf.reduce_mean(tf.squared_difference(y_pred, y))
                #batch_count+=1

        #averge_mse=total_mse/batch_count
        #PSNR=-10*math.log10(averge_mse)
        print("time: [%4.2f]" % (time.time()-start_time))
        
        #save
            #flatten
        print(len(result))
        output=list()
        for i in result:
            for j in range(i.shape[0]):
                output.append(i[j])
        print(len(output))
        print(output[0].shape)
        
        #result=[self.sess.run(i) for i in result]
        patch_inx=0
        for i in range(len(nxny_list)):
            nx,ny=nxny_list[i]
            img=merge(output[patch_inx:(patch_inx+nx*ny)],(nx,ny))
            print('img shape@',i,img.shape)
            patch_inx+=nx*ny
            imsave(img,namelist[i].replace('.bmp','.bmp.c'+str(self.config.c_dim)))
                    
    def train(self):
        #data preprocessing
        if(input_setup(self.sess, self.config)):#7-1-1
            print('generating patches...')
        else:
            print('found existing h5 files...')

        #build image path  
        trn_data_dir = os.path.join(self.config.checkpoint_dir,'train.c'+str(self.config.c_dim)+'.h5')
        print('trn_data_dir',trn_data_dir)
        X_train,y_train=read_data(trn_data_dir)
        trn_data_loader=dataLoader(dataSize=X_train.shape[0],
                                   batchSize=self.config.batch_size,
                                   shuffle=True,
                                   seed=345)
        
        tst_data_dir = os.path.join(self.config.checkpoint_dir,'test.c'+str(self.config.c_dim)+'.h5')
        print('tst_data_dir',tst_data_dir)
        X_test,y_test=read_data(tst_data_dir)#7-1-2 read image from h5py
        tst_data_loader=dataLoader(dataSize=X_test.shape[0],
                                   batchSize=self.config.test_batch_size,
                                   shuffle=False)
        
        #data description
        print('X_train.shape',X_train.shape)
        print('y_train.shape',y_train.shape)
        print('X_test.shape',X_test.shape)
        print('y_test.shape',y_test.shape)
        #del X_train,y_train,X_test,y_test
        #gc.collect()
  
        tf.global_variables_initializer().run()###remove DEPRECATED function###tf.initialize_all_variables().run()
    
        #Try to load pretrained model from checkpoint_dir
        if self.load(self.config.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        #if training
        # summary
            #loss and PSNR
        avg_trn_loss=tf.placeholder(tf.float32,shape=[],name='average_training_loss')
        tf.summary.scalar('average_training_loss',avg_trn_loss)
        
        avg_trn_psnr=tf.placeholder(tf.float32,shape=[],name='average_training_PSNR')
        tf.summary.scalar('average_training_PSNR',avg_trn_psnr)
        
        avg_tst_loss=tf.placeholder(tf.float32,shape=[],name='average_testing_loss')
        tf.summary.scalar('average_testing_loss',avg_tst_loss)
        
        avg_tst_psnr=tf.placeholder(tf.float32,shape=[],name='average_testing_PSNR')
        tf.summary.scalar('average_testing_PSNR',avg_tst_psnr)
        
            #weight and bias
#        w1_ph=tf.placeholder(tf.float32,shape=[9,9,self.config.c_dim,64],name='W1_ph')
#        b1_ph=tf.placeholder(tf.float32,shape=[64],name='b1_ph')
#        tf.summary.histogram('W1',w1_ph)
#        tf.summary.histogram('b1',b1_ph)
#        
#        w2_ph=tf.placeholder(tf.float32,shape=[5,5,64,32],name='W2_ph')
#        b2_ph=tf.placeholder(tf.float32,shape=[32],name='b2_ph')
#        tf.summary.histogram('W2',w2_ph)
#        tf.summary.histogram('b2',b2_ph)
#        
#        w3_ph=tf.placeholder(tf.float32,shape=[5,5,32,1],name='W3_ph')
#        b3_ph=tf.placeholder(tf.float32,shape=[1],name='b3_ph')
#        tf.summary.histogram('W3',w3_ph)
#        tf.summary.histogram('b3',b3_ph)
        
            #visualization
#        input_ph=tf.placeholder(tf.float32,shape=[self.config.batch_size,self.config.image_size, self.config.image_size,1],name='input_ph')#the center channel
#        tf.summary.image('sample_input',input_ph,3)
#        
#        output_ph=tf.placeholder(tf.float32,shape=[self.config.batch_size,self.config.label_size, self.config.label_size,1],name='output_ph')
#        tf.summary.image('sample_output',output_ph,3)
        
        merged_summary=tf.summary.merge_all()
        writer=tf.summary.FileWriter(self.config.TB_dir)
        writer.add_graph(self.sess.graph)
        
        print("Training...")
        batch_count=int(math.ceil(X_train.shape[0]/self.config.batch_size))
        tst_batch_count=int(math.ceil(X_test.shape[0]/self.config.test_batch_size))
        best_PSNR=0.
        best_ep=0.
        patience=self.config.patience

        for ep in range(self.config.epoch):#for each epoch
            epoch_loss = 0.
            start_time = time.time()
            for batch in range(batch_count):
                inx=trn_data_loader.get_batch()
                X,y = X_train[inx].view(),y_train[inx].view()
                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: X, self.labels: y})#update weights and biases 
                    #print('err',err)
                epoch_loss += err            
            trn_average_loss = epoch_loss / batch_count #per sample
            #print(self.sess.run(average_loss))
            trn_PSNR=-10*math.log10(trn_average_loss)
            print("Epoch: [%2d], \n\ttime: [%4.2f], \n\ttraining loss: [%.8f], \n\tPSNR: [%.4f]" % (ep, time.time()-start_time, trn_average_loss,trn_PSNR))
            
            #valid
            epoch_loss = 0.
            start_time = time.time()
            for batch in range(tst_batch_count):
                inx=tst_data_loader.get_batch()
                X,y = X_test[inx].view(),y_test[inx].view()
                err = self.sess.run(self.loss, feed_dict={self.images: X, self.labels: y})#only compute err
                epoch_loss += err
            tst_average_loss = epoch_loss / tst_batch_count #per sample
            tst_PSNR=-10*math.log10(tst_average_loss) 
            print("\n\ttime: [%4.2f], \n\ttesting loss: [%.8f], \n\tPSNR: [%.4f]\n\n" % (time.time()-start_time, tst_average_loss,tst_PSNR))
            
            #summary
            if ep%1==0:
                summ=self.sess.run(merged_summary,feed_dict={avg_trn_loss:trn_average_loss,
                                                             avg_trn_psnr:trn_PSNR,
                                                             avg_tst_loss:tst_average_loss,
                                                             avg_tst_psnr:tst_PSNR,
                                                             self.images:X,
                                                             self.labels:y
                                                             })
#                                                             w1_ph:self.weights['w1'],
#                                                             b1_ph:self.biases['b1'],
#                                                             w2_ph:self.weights['w2'],
#                                                             b2_ph:self.biases['b2'],
#                                                             w3_ph:self.weights['w3'],
#                                                             b3_ph:self.biases['b3'],
#                                                             input_ph:tf.reshape(self.images[:,:,:,(self.config.c_dim-1)//2],[self.config.batch_size,self.config.image_size,self.config.image_size,1]),
#                                                             output_ph:self.pred})
                writer.add_summary(summ,ep)
            #save
            if tst_PSNR<=best_PSNR:
                patience-=1
                if patience==0:
                    print('early stop!')
                    break
            else:# PSNR>best_PSNR:
                #print('\tcurrent best PSNR: <%.4f>\n' % PSNR)
                self.save(self.config.checkpoint_dir,ep)
                best_ep=ep
                best_PSNR=tst_PSNR
                patience=self.config.patience
        print('best ep',best_ep)
        print('best PSNR',best_PSNR)

    def save(self, checkpoint_dir, step):
        model_name = "CASRCNN_C"+str(self.config.c_dim)+".model"
        model_dir = "%s_%s_%s" % ("srcnn", self.config.label_size, self.config.c_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s_%s" % ("srcnn", self.config.model_label_size, self.config.c_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print('checkpoint_dir',checkpoint_dir)#print folder path out
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('model_checkpoint_path',ckpt.model_checkpoint_path)#model path
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            return self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            return False
