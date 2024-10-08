import torch
import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from skimage.transform import resize
from collections import OrderedDict
import matplotlib.pyplot as plt


def draw_result(epoch, y, title, path, y2=None):
    fig = plt.figure()
    plt.plot(epoch, y, 'r', label='val')
    if y2 is not None:
        plt.plot(epoch, y2, 'b', label='training')
    plt.xlabel("Epoch")
    plt.legend()
    plt.title(title)
    # save image
    plt.savefig(os.path.join(path, title + ".png"))  # should before show method
    plt.close(fig)


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            util.del_file(self.log_name)
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================Training Loss (%s) ================\n' % now)
            self.structure_name = os.path.join(opt.checkpoints_dir, opt.name, 'structure.txt')
            util.del_file(self.structure_name)
            self.metric_data = OrderedDict([('SSIM', []), ('MSSIM', [])])

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.opt.display_single_pane_ncols
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values

    def add_errors(self, errors):
        self.data_error = [errors[k].cpu().detach().numpy() + self.data_error[i] for i, k in enumerate(errors.keys())]

    def append_error_hist(self, total_iter, val=False):
        for i, leg in enumerate(self.plot_data['legend']):
            if not val:
                self.plot_data['train'][leg].append(self.data_error[i] / total_iter)
            else:
                self.plot_data['val'][leg].append(self.data_error[i] / total_iter)

    def plot_current_errors(self):
        for i, leg in enumerate(self.plot_data['legend']):
            y = [[k, l] for k, l in zip(self.plot_data['train'][leg], self.plot_data['val'][leg])]
            x = np.stack([np.array(range(len(y)))] * 2, 1)
            self.vis.line(
                X=x,
                Y=np.array(y),
                opts={
                    'title': leg + self.name + ' loss over time',
                    'legend': ['train', 'val'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id + i + 4)
            draw_result(np.array(range(len(y))),
                        [k for k in self.plot_data['val'][leg]],
                        leg,
                        self.web_dir,
                        [k for k in self.plot_data['train'][leg]])

    def plot_current_metrics(self, ssim):
        self.metric_data['SSIM'].append(ssim)
        y = np.array(self.metric_data['SSIM'])
        epoch = range(len(self.metric_data['SSIM']))
        self.vis.line(
            X=np.array(epoch),
            Y=y,
            opts={
                'title': self.name + ' metric time',
                'legend': ["SSIM"],
                'xlabel': 'epoch',
                'ylabel': 'similarity'},
            win=self.display_id)
        draw_result(epoch, self.metric_data['SSIM'], "SSIM", self.web_dir)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, t_data):
        message = '(epoch:%d iters:%d time:%.3f data:%.3f) ' % (epoch, i, t, t_data)
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def logger_structure(self, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        with open(self.structure_name, "a") as log_file:
            log_file.write(f'{net.__class__.__name__}: Total number of parameters: {num_params / 1e6}\n')

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, aspect_ratio=1.0):
        image_dir = webpage.get_image_dir()
        # short_path = ntpath.basename(image_path[0])
        # name = os.path.splitext(short_path)[0]
        name = image_path.split('/')[-1].split('.')[0].split('_')[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, im in visuals.items():
            image_name = '%s_%s.png' % (name, label)

            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = resize(im, (h, int(w * aspect_ratio)), order=3)  # interp='bicubic')
            if aspect_ratio < 1.0:
                im = resize(im, (int(h / aspect_ratio), w), order=3)  # interp='bicubic')
            util.save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
