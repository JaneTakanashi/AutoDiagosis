# -*- coding: utf-8 -*-
import tornado.ioloop
import tornado.web
import os
import json
from PIL import Image
import StringIO
import evaluate
caffe_root='/home/jane/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import matplotlib.pyplot as plt
import os
import pylab
import numpy as np

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("main_materialize.html")

class Submit(tornado.web.RequestHandler):
    def post(self):
        # get input tpye in FormData
        pre_file_body = self.request.files['pre'][0]['body']
        post_file_body = self.request.files['post'][0]['body']
        pre_img = Image.open(StringIO.StringIO(pre_file_body))
        post_img = Image.open(StringIO.StringIO(post_file_body))

        pre_img_x = int(float(self.get_body_argument('pre_img_x')))
        pre_img_y = int(float(self.get_body_argument('pre_img_y')))
        pre_img_width = int(float(self.get_body_argument('pre_img_width')))
        pre_img_height = int(float(self.get_body_argument('pre_img_height')))

        post_img_x = int(float(self.get_body_argument('post_img_x')))
        post_img_y = int(float(self.get_body_argument('post_img_y')))
        post_img_width = int(float(self.get_body_argument('post_img_width')))
        post_img_height = int(float(self.get_body_argument('post_img_height')))


        cropped_pre_img = pre_img.crop((pre_img_x, pre_img_y, pre_img_width+pre_img_x, pre_img_height+pre_img_y))
        cropped_post_img = post_img.crop((post_img_x, post_img_y, post_img_width+post_img_x, post_img_height+post_img_y))
        cropped_pre_img = cropped_pre_img.resize((150, 299))
        cropped_post_img = cropped_post_img.resize((149, 299))
        # cropped_pre_img.show()
        # cropped_post_img.show()

        cropped_pre_img.save('./pre.jpg')
        cropped_post_img.save('./post.jpg')

        img_combine = Image.new('RGB', (299,299),(255,255,255))
        img_combine.paste(cropped_pre_img,(0,0))
        img_combine.paste(cropped_post_img,(cropped_pre_img.width,0))
        img_combine.save('combine.jpg')

        MODEL_FILE = './static/net/train_val_changed_deploy.prototxt'
        PRETRAINED_re2 = './static/net/dentalnet_fullpath_re2__iter_20000.caffemodel'
        model = evaluate.CaffeModel(MODEL_FILE, PRETRAINED_re2)
        out = model.predict('combine.jpg')
        # print(self.get_argument('pre', None))
        pre = out.argmax()
        print('pre---------------')
        print pre
        treatment=['无变化','好转','恶化']
        self.write('<p> 本组牙片的疗效判定为：'+treatment[pre]+'<br />无变化的概率为：'+str(out[0][0])+'<br /> 好转的概率为：'+str(out[0][1])+'<br />恶化的概率为：'+str(out[0][2])+'</p>')





def make_app():
    # print os.path.dirname(__file__)

    # print os.path.join(os.path.dirname(__file__), "templates")
    return tornado.web.Application([
        (r"/",MainHandler),
        (r"/submit", Submit),
        ],
    static_path=os.path.join(os.path.dirname(__file__), "static"),
    template_path=os.path.join(os.path.dirname(__file__), "templates"),
    )

if __name__ == "__main__":
    app = make_app()
    app.listen(23333)

    tornado.ioloop.IOLoop.current().start()
