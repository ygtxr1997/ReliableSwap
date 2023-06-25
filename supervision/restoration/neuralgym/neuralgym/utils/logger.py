import logging
import shutil
import sys
import time


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def colored_log(prompt, texts, color='green', bold=True, highlight=False):
    """Show colored logs.
    """
    assert isinstance(prompt, str)
    assert isinstance(texts, str)
    assert isinstance(color, str)
    colored_prompt = colorize(prompt, color, bold=bold, highlight=highlight)
    clean_line = ''
    sys.stdout.write(clean_line)
    print(colored_prompt + texts)


def callback_log(texts):
    """Callback_log will show caller's location.

    Args:
        texts (str): Text to show.

    """
    colored_log('Trigger callback: ', texts)


def warning_log(texts):
    """Warning_log will show caller's location and red texts.

    Args:
        texts (str): Text to show.

    """
    colored_log('Warning: ', texts, color='red')


def error_log(texts):
    """Error_log will show caller's location, red texts and raise
    RuntimeError.

    Args:
        texts (str): Text to show.

    """
    colored_log('Error: ', texts, color='red')
    raise RuntimeError


class ProgressBar(object):
    """ Visualize progress.

    It displays a progress bar in console with time recorder and statistics.
    """

    def __init__(self):
        # length of progress bar = terminal width / split_n
        self._split_n = 4
        self._t_start = None
        self._t_last = None
        self._t_current = None
        # progress recorders
        self._p_start = None
        self._p_last = None
        self._p_current = None
        # restart at init
        self.restart()

    def restart(self):
        """Restart time recorder and progress recorder.

        """
        # time recorders
        self._t_start = time.time()
        self._t_last = self._t_start
        self._t_current = self._t_start
        # progress recorders
        self._p_start = 0
        self._p_last = self._p_start
        self._p_current = self._p_start

    def progress(self, progress, texts=''):
        """Update progress bar with current progress and additional texts.

        Args:
            progress (float): A float between [0,1] indicating progress.
            texts (str): additional texts (e.g. statistics) appear at the end
                of progress bar.

        """
        term_length, _ = shutil.get_terminal_size()
        length = int(term_length / self._split_n)
        if isinstance(progress, int):
            progress = float(progress)
        assert isinstance(progress, float)
        assert isinstance(texts, str)
        assert progress >= 0 and progress <= 1, 'Progress is between [0,1].'
        # the number of '#' to be shown
        block = int(round(length*progress))
        # compute time and progress
        self._p_current = progress
        self._t_current = time.time()
        speed = (self._p_current-self._p_last)/(self._t_current-self._t_last)
        t_consumed = self._t_current - self._t_start
        t_remained = (1-self._p_current) / speed if speed != 0 else 0
        # info to be shown
        info = ''.join([
            '\x1b[2K\r|',
            '#'*block,
            '-'*(length-block),
            '|',
            ' {:.2f}%,'.format(self._p_current*100),
            ' {:.0f}/{:.0f} sec.'.format(t_consumed, t_remained),
            ' {}'.format(texts)
        ])

        if progress == 1:
            print(texts)
            self.restart()
        else:
            # sys.stdout.write(info)
            # update time and progress
            self._p_last = self._p_current
            self._t_last = self._t_current
