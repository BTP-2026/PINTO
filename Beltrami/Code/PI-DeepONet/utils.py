import numpy as np
from pyDOE import lhs


def get_ibc_and_inner_data(start, stop, init_samples, bound_points, domain_samples, sensor_samples, re):
    lower_bound = np.array(start).reshape((1, -1))
    upper_bound = np.array(stop).reshape((1, -1))

    x_dom = (upper_bound - lower_bound) * lhs(3, domain_samples) + lower_bound

    # left boundary points
    yt_left = (upper_bound - lower_bound)[0:, 1:] * lhs(2, bound_points) + lower_bound[0:, 1:]
    x_left = np.ones((bound_points, 1)) * lower_bound[0, 0]
    bound_left = np.concatenate((x_left, yt_left), axis=1)

    # right boundary points
    yt_right = (upper_bound - lower_bound)[0:, 1:] * lhs(2, bound_points) + lower_bound[0:, 1:]
    x_right = np.ones((bound_points, 1)) * upper_bound[0, 0]
    bound_right = np.concatenate((x_right, yt_right), axis=1)

    # top boundary points
    xt_top = (upper_bound - lower_bound)[0:, 0::2] * lhs(2, bound_points) + lower_bound[0:, 0::2]
    y_top = np.ones((bound_points, 1)) * upper_bound[0, 1]
    bound_top = np.concatenate((xt_top[:, 0:1], y_top, xt_top[:, 1:]), axis=1)

    # bottom boundary points
    xt_bottom = (upper_bound - lower_bound)[0:, 0::2] * lhs(2, bound_points) + lower_bound[0:, 0::2]
    y_bottom = np.ones((bound_points, 1)) * lower_bound[0, 1]
    bound_bottom = np.concatenate((xt_bottom[:, 0:1], y_bottom, xt_bottom[:, 1:]), axis=1)

    # Initial Condition
    xy_init = (upper_bound - lower_bound)[0:, :2] * lhs(2, init_samples) + lower_bound[0:, :2]
    t_init = np.ones((init_samples, 1)) * lower_bound[0, 2]
    bound_init = np.concatenate((xy_init, t_init), axis=1)

    x_bound = np.concatenate((bound_init, bound_left, bound_right, bound_top, bound_bottom), axis=0)

    # print(f"rank of x_dom should be 2 {x_dom.shape}")
    # print(f"rank of x_bound should be 2 {x_bound.shape}")

    Xe_dom = np.repeat(np.expand_dims(x_dom, axis=0), len(re), axis=0)
    nue_d = np.ones_like(Xe_dom[:, :, 0:1]) * (np.array(re).reshape((-1, 1, 1)))
    Xe_bound = np.repeat(np.expand_dims(x_bound, axis=0), len(re), axis=0)
    nue_b = np.ones_like(Xe_bound[:, :, 0:1]) * (np.array(re).reshape((-1, 1, 1)))

    # print(f"rank of Xe_dom should be 3 {Xe_dom.shape}")
    # print(f"rank of Xe_bound should be 3 {Xe_bound.shape}")

    xd = Xe_dom[:, :, 0:1]
    assert (xd.shape == nue_d.shape)
    yd = Xe_dom[:, :, 1:2]
    assert (yd.shape == nue_d.shape)
    td = Xe_dom[:, :, 2:]
    assert (td.shape == nue_d.shape)

    xb = Xe_bound[:, :, 0:1]
    assert (xb.shape == nue_b.shape)
    yb = Xe_bound[:, :, 1:2]
    assert (yb.shape == nue_b.shape)
    tb = Xe_bound[:, :, 2:]
    assert (tb.shape == nue_b.shape)

    u_bound, v_bound, p_bound = get_fvalues(xb, yb, tb, nue_b)

    # Sensor Data
    # left boundary points
    yt_ls = (upper_bound - lower_bound)[0:, 1:] * lhs(2, sensor_samples) + lower_bound[0:, 1:]
    x_ls = np.ones((sensor_samples, 1)) * lower_bound[0, 0]
    sen_left = np.concatenate((x_ls, yt_ls), axis=1)

    # right boundary points
    yt_rs = (upper_bound - lower_bound)[0:, 1:] * lhs(2, sensor_samples) + lower_bound[0:, 1:]
    x_rs = np.ones((sensor_samples, 1)) * upper_bound[0, 0]
    sen_right = np.concatenate((x_rs, yt_rs), axis=1)

    # top boundary points
    xt_ts = (upper_bound - lower_bound)[0:, 0::2] * lhs(2, sensor_samples) + lower_bound[0:, 0::2]
    y_ts = np.ones((sensor_samples, 1)) * upper_bound[0, 1]
    sen_top = np.concatenate((xt_ts[:, 0:1], y_ts, xt_ts[:, 1:]), axis=1)

    # bottom boundary points
    xt_bs = (upper_bound - lower_bound)[0:, 0::2] * lhs(2, sensor_samples) + lower_bound[0:, 0::2]
    y_bs = np.ones((sensor_samples, 1)) * lower_bound[0, 1]
    sen_bottom = np.concatenate((xt_bs[:, 0:1], y_bs, xt_bs[:, 1:]), axis=1)

    # Initial Condition
    xy_is = (upper_bound - lower_bound)[0:, :2] * lhs(2, sensor_samples) + lower_bound[0:, :2]
    t_is = np.ones((sensor_samples, 1)) * lower_bound[0, 2]
    sen_init = np.concatenate((xy_is, t_is), axis=1)

    X_sen = np.concatenate((sen_init, sen_left, sen_right, sen_top, sen_bottom), axis=0)
    Xe_sen = np.repeat(np.expand_dims(X_sen, axis=0), len(re), axis=0)
    # print(f"rank of Xe_sen should be 3 {Xe_sen.shape}")
    nue_sen = np.ones_like(Xe_sen[:, :, 0:1]) * (np.array(re).reshape((-1, 1, 1)))
    x_sen = Xe_sen[:, :, 0:1]
    assert (x_sen.shape == nue_sen.shape)
    y_sen = Xe_sen[:, :, 1:2]
    assert (y_sen.shape == nue_sen.shape)
    t_sen = Xe_sen[:, :, 2:]
    assert (t_sen.shape == nue_sen.shape)
    u_sensor, v_sensor, p_sensor = get_fvalues(x_sen, y_sen, t_sen, nue_sen)

    x_sensor = np.transpose(x_sen, [0, 2, 1])
    y_sensor = np.transpose(y_sen, [0, 2, 1])
    t_sensor = np.transpose(t_sen, [0, 2, 1])
    u_sensor = np.transpose(u_sensor, [0, 2, 1])
    v_sensor = np.transpose(v_sensor, [0, 2, 1])
    p_sensor = np.transpose(p_sensor, [0, 2, 1])

    ins = xd.shape[1]
    bs = xb.shape[1]

    xbc_in = np.repeat(x_sensor, ins, axis=1).reshape((-1, 5 * sensor_samples))
    ybc_in = np.repeat(y_sensor, ins, axis=1).reshape((-1, 5 * sensor_samples))
    tbc_in = np.repeat(t_sensor, ins, axis=1).reshape((-1, 5 * sensor_samples))
    ubc_in = np.repeat(u_sensor, ins, axis=1).reshape((-1, 5 * sensor_samples))
    vbc_in = np.repeat(v_sensor, ins, axis=1).reshape((-1, 5 * sensor_samples))
    pbc_in = np.repeat(p_sensor, ins, axis=1).reshape((-1, 5 * sensor_samples))

    xbc_b = np.repeat(x_sensor, bs, axis=1).reshape((-1, 5 * sensor_samples))
    ybc_b = np.repeat(y_sensor, bs, axis=1).reshape((-1, 5 * sensor_samples))
    tbc_b = np.repeat(t_sensor, bs, axis=1).reshape((-1, 5 * sensor_samples))
    ubc_b = np.repeat(u_sensor, bs, axis=1).reshape((-1, 5 * sensor_samples))
    vbc_b = np.repeat(v_sensor, bs, axis=1).reshape((-1, 5 * sensor_samples))
    pbc_b = np.repeat(p_sensor, bs, axis=1).reshape((-1, 5 * sensor_samples))

    return (xd.reshape((-1, 1)), yd.reshape((-1, 1)), td.reshape((-1, 1)),
            xb.reshape((-1, 1)), yb.reshape((-1, 1)), tb.reshape((-1, 1)), u_bound.reshape((-1, 1)),
            v_bound.reshape((-1, 1)), p_bound.reshape((-1, 1)), xbc_in, ybc_in, tbc_in, ubc_in, vbc_in, pbc_in,
            xbc_b, ybc_b, tbc_b, ubc_b, vbc_b, pbc_b, nue_d.reshape((-1, 1)), X_sen)


def get_fvalues(X, Y, T, nue):
    u_val = -np.cos(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2 * np.pi ** 2 * nue * T)
    v_val = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.exp(-2 * np.pi ** 2 * nue * T)
    p_val = (-((np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)) / 4) *
             np.exp(-4 * np.pi ** 2 * nue * T))

    # print(f"rank of u_val, v_val, p_val 3 with last index 1: {u_val.shape, v_val.shape, p_val.shape}")

    return u_val, v_val, p_val
