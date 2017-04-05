# from time import sleep


def load_data(*args, **kwargs):
    # sleep(2)
    print 'load data ...'
    if args:
        print args
        return 'data_{}'.format(args)
    if kwargs:
        print kwargs
        return 'data_{}'.format(kwargs)
    return 'data'


def clean_data(data, *args, **kwargs):
    assert isinstance(data, str)
    print 'clean data ...'
    if args:
        print args
        data = data + '_' + '_'.join(map(lambda x: '{}'.format(x), args))
    if kwargs:
        print kwargs
        data = data + '_' + \
            '_'.join(map(lambda kv: '{}_{}'.format(
                kv[0], kv[1]), kwargs.items()))
    return 'cleaned_{}'.format(data)


def analyze_data(cleaned_data, option=1, **other_options):
    assert isinstance(cleaned_data, str)
    print 'analyze data ...'
    return 'analyzed_{}'.format(cleaned_data)
