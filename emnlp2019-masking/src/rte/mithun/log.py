import logging

def setup_custom_logger(name, lmode,log_file_name):
    log_mode=logging.DEBUG

    if(lmode=="DEBUG"):
        log_mode = logging.DEBUG
    else:

        if (lmode == "WARNING"):
            log_mode = logging.WARNING

        else:

            if (lmode == "INFO"):
                log_mode = logging.INFO

            else:

                if (lmode == "ERROR"):
                    log_mode = logging.ERROR

    logging.basicConfig(level=log_mode,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=log_file_name,
                    filemode='w')

    logger = logging.getLogger(name)
    '''critical, error > warning,info, debug'''

    ch=logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    logging.getLogger('').addHandler(ch)



    return logger
