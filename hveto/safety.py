from __future__ import (division, print_function)
from hveto.core import (coinc_significance, find_all_coincidences,
                        HvetoWinner, significance, find_coincidences,
                        HvetoRound, veto)
from hveto import (__version__, log, config, core, plot, html, utils)
from hveto.triggers import (get_triggers, find_auxiliary_channels,
                            get_column_label)
from gwpy.segments import (Segment, SegmentList,
                           DataQualityFlag, DataQualityDict)
from astropy.table import vstack as vstack_tables
import math
from astropy.time import Time
from numpy import unique
from safety_html import (write_hveto_safety_page, write_safety_round)

import collections
import os
import multiprocessing
import subprocess


def safety_calc(cp, logger, args,
                primary, auxiliary, channel, snrs, windows, livetime,
                outdir, plotdir, omegadir, inifile, unsafe,
                safety_cutoff, safety_plot_count,
                ifo, start, end, analysis, jobstart):
    """
    Calculate significance for each channel
    :param cp: configuration parameters
    :param logger: logging object
    :param args: command line arguments

    :param primary:
    :param auxiliary:
    :param channel:
    :param snrs:
    :param windows:
    :param livetme

    :param outdir: html, configs ...
    :param plotdir: our gwpy generated plots
    :param omegadir: Wdq output
    :param inifile: path to config file used
    :param unsafe: set of unsafe channel names

    :param safety_cutoff:
    :param safety_plot_count:

    :param ifo:
    :param start:
    :param end:
    :param analysis: segment used
    :param jobstart: start time for reporting
    :return: stat [true if successful]
    """
    stat = False
    safety_cutoff = cp.getfloat('safety', 'safety_cutoff') # >= cutoff -> unsafe
    safety_min = cp.getfloat('safety', 'safety_min')        # plot if significance >= min

    logger.debug('Safety params: safety_cutoff: %.2f, safety_min: %d' %
                 (safety_cutoff, safety_min))

    fcol, scol = primary.dtype.names[1:3]

    auxchannels =  auxiliary.keys()
    auxfcol, auxscol = auxiliary[auxchannels[0]].dtype.names[1:3]
    slabel = get_column_label(scol)
    flabel = get_column_label(fcol)
    auxslabel = get_column_label(auxscol)
    auxflabel = get_column_label(auxfcol)

    petg = cp.get('primary', 'trigger-generator')
    psnr = cp.getfloat('primary', 'snr-threshold')
    pfreq = cp.getfloats('primary', 'frequency-range')

    chan_num = 0


    rec = vstack_tables([primary] + auxiliary.values())
    coincs = find_all_coincidences(rec, channel, snrs, windows)

    chan_stats = []
    rounds = []
    duration = end - start

    sigs = dict((c, 0) for c in auxiliary)
    for p, cdict in coincs.items():
        dt, snr = p
        for chan in cdict:
            mu = (len(primary) * (auxiliary[chan]['snr'] >= snr).sum() *
                  dt / livetime)
            # NOTE: coincs[p][chan] counts the number of primary channel
            # triggers coincident with a 'chan' trigger
            try:
                allaux = auxiliary[chan][auxiliary[chan][scol] >= snr]
                coinc1 = allaux[find_coincidences(allaux['time'], primary['time'], dt=dt)]

                trig_cnt = len(coinc1)
                sig = significance(trig_cnt, mu)
            except KeyError:
                sig == 0

            if trig_cnt > 0:
                # save vals for every channel. here everyone is a winner
                chan_stat = HvetoWinner(name=chan)

                chan_stat.snr = snr
                chan_stat.window = dt
                chan_stat.significance = sig
                chan_stat.mu = mu
                chan_stat.ncoinc = trig_cnt

                chan_stats.append(chan_stat)

    # sort by significance
    chan_stats = sorted(chan_stats, key=lambda x: x.significance, reverse=True)

    # ascii results
    sum_file = outdir + "/safety_summary.csv"


    with open(sum_file, 'w') as f:
        # CSV column name
        vals = 'chan, significance, N-coinc, N-expected, '
        vals += 'N-aux, N-coinc/N-expected, SNR, dT '
        print(vals, file=f)

        for cs in chan_stats:
            name = cs.name
            allaux = auxiliary[name][auxiliary[name][scol] >= cs.snr]
            coincs = allaux[find_coincidences(allaux['time'], primary['time'], dt=dt)]

            # recalc because of bug(?) in find_all_coincidences
            try:
                trig_cnt = len(coincs)
                sig = significance(trig_cnt, cs.mu)
            except KeyError:
                sig == 0

            if sig > safety_min:
                cs.ncoinc = trig_cnt
                cs.significance = sig
                chan_num += 1

                dups = [item for item, count in collections.Counter(coincs['time']).items() if count > 1]
                dup_str = ''
                if len(dups) > 0:
                    dup_str = ','.join(map(str,coincs['gpstime']))

                vals = '%s,%.3f,%d,%.3f,%.0f,%.4f,%.1f,%.2f,%s' %\
                       (name, cs.significance, cs.ncoinc, cs.mu,
                        len(allaux), cs.ncoinc / cs.mu, cs.snr, cs.window, dup_str)
                print(vals, file=f)

                # process as if it were a round winner
                if cs.significance > safety_min:
                    round = HvetoRound(chan_num, primary)
                    round.segments = analysis.active
                    # work out the vetoes for this round

                    cs.events = allaux
                    round.vetoes = cs.get_segments(allaux['time'])
                    flag = DataQualityFlag(
                        '%s:HVT-CHAN_%d:1' % (ifo, round.n), active=round.vetoes,
                        known=round.segments,
                        description="winner=%s, window=%s, snr=%s" % (
                            cs.name, cs.window, cs.snr))
                    before = primary
                    beforeaux = auxiliary[cs.name]

                    # apply vetoes to primary
                    primary_vetoed, vetoed = veto(primary, round.vetoes)

                    aux_vetoed, avetoed = veto(beforeaux, round.vetoes)

                    round.winner = cs
                    round.efficiency = (len(vetoed), len(primary_vetoed) + len(vetoed))
                    round.use_percentage = (len(coincs), len(cs.events))
                    round.cum_efficiency = round.efficiency
                    round.cum_deadtime = round.deadtime

                    logger.debug('%4d. %s ' % (round.n, cs.name))
                    # log results
                    logger.info("""Results for round %d
winner :          %s
significance :    %s
mu :              %s
snr :             %s
dt :              %s
use_percentage :  %s
efficiency :      %s
deadtime :        %s
cum. efficiency : %s
cum. deadtime :   %s""" % (
                        round.n, round.winner.name, round.winner.significance, round.winner.mu,
                        round.winner.snr, round.winner.window, round.use_percentage,
                        round.efficiency, round.deadtime, round.cum_efficiency,
                        round.cum_deadtime))

                    # record times for omega scans
                    if args.omega_scans:
                        vetoed.sort(scol)
                        round.scans = vetoed[-args.omega_scans:][::-1]
                        logger.debug("Identified %d events for omega scan\n%s"
                                     % (args.omega_scans, round.scans))

                    pngname = os.path.join(plotdir, '%s-HVETO_%%s_SAFETY_ROUND_%03d-%d-%d.png' %
                                           (ifo, round.n, start, duration))
                    if plot.rcParams['text.usetex']:
                        wname = round.winner.name.replace('_', r'\_')
                    else:
                        wname = round.winner.name
                    beforel = 'Before\n[%d]' % len(before)
                    afterl = 'After\n[%d]' % len(primary_vetoed)
                    vetoedl = 'Vetoed\n[%d]' % len(vetoed)
                    beforeauxl = 'All\n[%d]' % len(beforeaux)
                    usedl = 'Used\n[%d]' % len(cs.events)
                    coincl = 'Coinc.\n[%d]' % len(coincs)
                    title = '%s Hveto round %d, significance %.1f' % (
                        ifo, round.n, round.winner.significance)
                    ptitle = '%s: primary impact' % title
                    atitle = '%s: auxiliary use' % title
                    subtitle = '[%d-%d] | winner: %s' % (start, end, wname)

                    # before/after histogram
                    png = pngname % 'HISTOGRAM'
                    if not os.path.isfile(png):
                        plot.before_after_histogram(
                            png, before[scol], primary[scol],
                            label1=beforel, label2=afterl, xlabel=slabel,
                            title=ptitle, subtitle=subtitle)
                        logger.debug("Figure written to %s" % png)
                    round.plots.append(png)

                    # snr versus time
                    png = pngname % 'SNR_TIME'
                    if not os.path.isfile(png):
                        plot.veto_scatter(
                            png, before, vetoed, x='time', y=scol, label1=beforel, label2=vetoedl,
                            epoch=start, xlim=[start, end], ylabel=slabel,
                            title=ptitle, subtitle=subtitle, legend_title="Primary:")
                        logger.debug("Figure written to %s" % png)
                    round.plots.append(png)

                    # snr versus frequency
                    png = pngname % 'SNR_%s' % fcol.upper()
                    if not os.path.isfile(png):
                        plot.veto_scatter(
                            png, before, vetoed, x=fcol, y=scol, label1=beforel,
                            label2=vetoedl, xlabel=flabel, ylabel=slabel, xlim=pfreq,
                            title=ptitle, subtitle=subtitle, legend_title="Primary:")
                        logger.debug("Figure written to %s" % png)
                    round.plots.append(png)

                    # frequency versus time coloured by SNR
                    png = pngname % '%s_TIME' % fcol.upper()
                    if not os.path.isfile(png):
                        plot.veto_scatter(
                            png, before, vetoed, x='time', y=fcol, color=scol,
                            label1=None, label2=None, ylabel=flabel,
                            clabel=slabel, clim=[3, 100], cmap='YlGnBu',
                            epoch=start, xlim=[start, end], ylim=pfreq,
                            title=ptitle, subtitle=subtitle)
                        logger.debug("Figure written to %s" % png)
                    round.plots.append(png)

                    # aux used versus frequency
                    png = pngname % 'USED_SNR_TIME'
                    if not os.path.isfile(png):
                        plot.veto_scatter(
                            png, cs.events, vetoed, x='time', y=[auxscol, scol], label1=usedl,
                            label2=vetoedl, ylabel=slabel, epoch=start, xlim=[start, end],
                            title=atitle, subtitle=subtitle)
                        logger.debug("Figure written to %s" % png)
                    round.plots.append(png)

                    # snr versus time
                    png = pngname % 'AUX_SNR_TIME'
                    if not os.path.isfile(png):
                        plot.veto_scatter(
                            png, beforeaux, (cs.events, coincs), x='time', y=auxscol,
                            label1=beforeauxl, label2=(usedl, coincl), epoch=start,
                            xlim=[start, end], ylabel=auxslabel, title=atitle, subtitle=subtitle)
                        logger.debug("Figure written to %s" % png)
                    round.plots.append(png)

                    # snr versus frequency
                    png = pngname % 'AUX_SNR_FREQUENCY'
                    if not os.path.isfile(png):
                        plot.veto_scatter(
                            png, beforeaux, (cs.events, coincs), x=auxfcol, y=auxscol,
                            label1=beforeauxl, label2=(usedl, coincl), xlabel=auxflabel,
                            ylabel=auxslabel, title=atitle, subtitle=subtitle, legend_title="Aux:")
                        logger.debug("Figure written to %s" % png)
                    round.plots.append(png)

                    # frequency versus time coloured by SNR
                    png = pngname % 'AUX_FREQUENCY_TIME'
                    if not os.path.isfile(png):
                        plot.veto_scatter(
                            png, beforeaux, (cs.events, coincs), x='time', y=auxfcol,
                            color=auxscol, label1=None, label2=[None, None], ylabel=auxflabel,
                            clabel=auxslabel, clim=[3, 100], cmap='YlGnBu', epoch=start,
                            xlim=[start, end], title=atitle, subtitle=subtitle)
                        logger.debug("Figure written to %s" % png)
                    round.plots.append(png)

                    round.unsafe = round.winner.name in unsafe

                    rounds.append(round)

    # -- write HTML
    # prepare html variables
    startUTC = Time(start, format='gps', scale='utc')
    endUTC = Time(end, format='gps', scale='utc')
    timeStr = '%s - %s' % (startUTC.iso, endUTC.iso)

    title_txt = '%s Hveto Safety  %d-%d (%s)' % (ifo, start, end, timeStr)

    htmlv = {
        'title': title_txt,
        'config': None,
    }
    htmlv['config'] = inifile

    index = write_hveto_safety_page(ifo, start, end, rounds, safety_cutoff, **htmlv)
    logger.debug("HTML written to %s" % index)

    for r in rounds:
        if r.winner.significance >= safety_min:
            htmlv['base'] = os.path.curdir
            htmlv['html_file'] = 'hveto-round-%04d-summary.html' % r.n
            write_safety_round(ifo, start, end, r, safety_cutoff, **htmlv)


    # -- generate workflow for omega scans
    if args.omega_scans:
        omegatimes = list(map(str, sorted(unique(
            [t['time'] for r in rounds for t in r.scans]))))
        logger.debug("Identified %d times for omega scan" % len(omegatimes))
        newtimes = [t for t in omegatimes if not
                    utils.omega_scan_complete(os.path.join(omegadir, str(t)))]
        logger.debug("%d scans already complete, %d remaining"
                     % (len(omegatimes) - len(newtimes), len(newtimes)))

        omega_time_file = os.path.join(outdir, 'omega_times.txt')
        with open(omega_time_file, 'w') as ot_file:
            for t in newtimes:
                ot_file.write('%s\n' % t)

        logger.info('Creating workflow for omega scans...')
        batch = ['wdq-batch', omega_time_file,
            '--output-dir', omegadir,
            '--ifo', ifo, '--wpipeline', '/home/omega/opt/omega/bin/wpipeline',
        ]
        proc = subprocess.Popen(batch)
        out, err = proc.communicate()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, ' '.join(batch))
        dagfile = os.path.join(omegadir, 'condor', 'wdq-batch.dag')

    import time
    run_time = time.time() - jobstart
    logger.info ('Safety runtime  %.1f sec' % run_time)

    return stat
