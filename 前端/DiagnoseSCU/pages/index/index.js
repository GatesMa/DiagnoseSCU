//index.js
//获取应用实例
const app = getApp()
const ip = app.ip
Page({
    data: {
        motto: 'Hello World',
        userInfo: {},
        hasUserInfo: false,
        canIUse: wx.canIUse('button.open-type.getUserInfo'),
        id: null,
        img1: null,
        img2: null,
        video1: null,
        video2: null,
        level:  null
    },
    //事件处理函数
    bindViewTap: function() {
        wx.navigateTo({
            url: '../logs/logs'
        })
    },
    onLoad: function() {
        if (app.globalData.userInfo) {
            this.setData({
                userInfo: app.globalData.userInfo,
                hasUserInfo: true
            })
        } else if (this.data.canIUse) {
            // 由于 getUserInfo 是网络请求，可能会在 Page.onLoad 之后才返回
            // 所以此处加入 callback 以防止这种情况
            app.userInfoReadyCallback = res => {
                this.setData({
                    userInfo: res.userInfo,
                    hasUserInfo: true
                })
            }
        } else {
            // 在没有 open-type=getUserInfo 版本的兼容处理
            wx.getUserInfo({
                success: res => {
                    app.globalData.userInfo = res.userInfo
                    this.setData({
                        userInfo: res.userInfo,
                        hasUserInfo: true
                    })
                }
            })
        }
    },

    getUserInfo: function(e) {
        console.log(e)
        app.globalData.userInfo = e.detail.userInfo
        this.setData({
            userInfo: e.detail.userInfo,
            hasUserInfo: true
        })
    },

    get_number: function() {
        var that = this
        if (that.data.id != null) {
            console.log("获取的ID：" + that.data.id)
            wx.showToast({
                title: '你已经获取了ID，现在可以上传数据诊断了',
                icon: 'none',
                duration: 1000
            })
            return
        }
        console.log("获取ID中...")
        wx.request({
            url: ip + '/get_number',
            method: 'GET',
            data: {
                
            },
            header: {
                'content-type': 'application/json' // 默认值
            },
            success(res) {
                console.log(res.data)
                that.setData({
                    id: res.data
                })
                wx.showToast({
                    title: '获取ID成功',
                    icon: 'success',
                    duration: 1000
                })
            },
            fail(err) {
                console.log('err :' + err.errMsg)
            },
            complete() {
                // console.log(type(res.data))
                console.log(ip + '/get_number')
                console.log('id: ' + that.data.id)
            }
        })
    },


    chooseImage1: function () {
        var that = this;
        if(that.data.id == null) {
            wx.showToast({
                title: '请先获取您的唯一ID',
                icon: 'none',
                duration: 1000
            })
            return
        }
        wx.chooseImage({
            count: 1,
            sizeType: ['original', 'compressed'],
            sourceType: ['album', 'camera'],
            success(res) {
                // tempFilePatxh可以作为img标签的src属性显示图片
                const tempFilePaths = res.tempFilePaths[0]
                console.log(tempFilePaths[0])
                that.setData({
                    userHeaderImage: tempFilePaths
                })
                console.log(tempFilePaths)
                //上传图片
                wx.showLoading({
                    title: '上传图片1中...',
                })
                wx.uploadFile({
                    url: ip + '/upload_img1', //仅为示例，非真实的接口地址
                    filePath: tempFilePaths,
                    name: 'file',
                    // header: {
                    //   "content-type": "multipart/form-data",
                    //   'content-type': 'application/x-www-form-urlencoded' //表单提交
                    // },
                    header: { "content-type": "multipart/form-data"},
                    formData: {
                        'id': that.data.id
                    },
                    success(res) {
                        const data = res.data
                        console.log(res);
                        console.log('upload_img1:' + res.data)
                        that.setData({
                            img1: true
                        })
                        //do something
                    },
                    complete() {
                        wx.hideLoading()
                    }
                })
            }, 
            fail(err) {
                console.log("img1上传失败：" + err.errMsg)
            }
        })
    },


    chooseImage2: function () {
        var that = this;
        if (that.data.id == null) {
            wx.showToast({
                title: '请先获取您的唯一ID',
                icon: 'none',
                duration: 1000
            })
            return
        }
        wx.chooseImage({
            count: 1,
            sizeType: ['original', 'compressed'],
            sourceType: ['album', 'camera'],
            success(res) {
                // tempFilePath可以作为img标签的src属性显示图片
                const tempFilePaths = res.tempFilePaths[0]
                console.log(tempFilePaths[0])
                that.setData({
                    userHeaderImage: tempFilePaths
                })
                console.log(tempFilePaths)
                //上传图片
                wx.showLoading({
                    title: '上传图片2中...',
                })
                wx.uploadFile({
                    url: ip + '/upload_img2', //仅为示例，非真实的接口地址
                    filePath: tempFilePaths,
                    name: 'file',
                    // header: {
                    //   "content-type": "multipart/form-data",
                    //   'content-type': 'application/x-www-form-urlencoded' //表单提交
                    // },
                    header: { "content-type": "multipart/form-data" },
                    formData: {
                        "id": that.data.id
                    },
                    success(res) {
                        const data = res.data
                        console.log(res);
                        that.setData({
                            img2: true
                        })
                        //do something
                    },
                    complete() {
                        wx.hideLoading()
                    }
                })
            }
        })
    },


    chooseVideo1: function() {
        var that = this
        if (that.data.id == null) {
            wx.showToast({
                title: '请先获取您的唯一ID',
                icon: 'none',
                duration: 1000
            })
            return
        }
        wx.chooseVideo({
            sourceType: ['album', 'camera'],
            maxDuration: 20,
            camera: 'back',
            success(res) {
                console.log('tempFilePath:' + res.tempFilePath)
                // tempFilePath可以作为img标签的src属性显示图片
                
                that.setData({
                    src: res.tempFilePath,
                })
                //上传视频
                wx.showLoading({
                    title: '上传视频1中...',
                })
                var src = that.data.src;
                wx.uploadFile({
                    url: ip + '/upload_video1', //仅为示例，非真实的接口地址
                    // filePath: tempFilePaths,
                    name: 'file',
                    // header: {
                    //   "content-type": "multipart/form-data",
                    //   'content-type': 'application/x-www-form-urlencoded' //表单提交
                    // },
                    filePath: src,
                    header: { "content-type": "multipart/form-data" },
                    formData: {
                        "id": that.data.id
                    },
                    success(res) {
                        const data = res.data
                        console.log(res);
                        that.setData({
                            video1: true
                        })
                        //do something
                        console.log('视频上传成功')
                    },
                    fail(err) {
                        console.log("upload_video1: error:" + err.errMsg)
                        console.log('接口调用失败')
                    },
                    complete() {
                        wx.hideLoading()
                    }
                })
            },
            fail(err) {
                console.log(err)
            }
        })
    },

    chooseVideo2: function () {
        var that = this
        if (that.data.id == null) {
            wx.showToast({
                title: '请先获取您的唯一ID',
                icon: 'none',
                duration: 1000
            })
            return
        }
        wx.chooseVideo({
            sourceType: ['album', 'camera'],
            maxDuration: 20,
            camera: 'back',
            success(res) {
                console.log('tempFilePath:' + res.tempFilePath)
                // tempFilePath可以作为img标签的src属性显示图片
                that.setData({
                    src: res.tempFilePath,
                })
                //上传视频
                wx.showLoading({
                    title: '上传视频2中...',
                })
                var src = that.data.src;
                wx.uploadFile({
                    url: ip + '/upload_video2', //仅为示例，非真实的接口地址
                    // filePath: tempFilePaths,
                    name: 'file',
                    // header: {
                    //   "content-type": "multipart/form-data",
                    //   'content-type': 'application/x-www-form-urlencoded' //表单提交
                    // },
                    filePath: src,
                    header: { "content-type": "multipart/form-data" },
                    formData: {
                        "id": that.data.id
                    },
                    success(res) {
                        const data = res.data
                        console.log(res);
                        that.setData({
                            video2: true
                        })
                        //do something
                        console.log('视频上传成功')
                    },
                    fail(err) {
                        console.log("upload_video1: error:" + err.errMsg)
                        console.log('接口调用失败')
                    },
                    complete() {
                        wx.hideLoading()
                    }
                })
            },
            fail(err) {
                console.log(err)
            }
        })
    },
    getresult: function() {
        var that = this
        if (that.data.id == null) {
            wx.showToast({
                title: '请先获取您的唯一ID',
                icon: 'none',
                duration: 1000
            })
            return
        }
        wx.showLoading({
            title: '获取处理结果',
        })
        console.log('获取处理结果')
        wx.request({
            url: ip + '/get_level/' + that.data.id, //仅为示例，并非真实的接口地址
            data: {

            },
            header: {
                'content-type': 'application/json' // 默认值
            },
            success(res) {
                console.log(res.data)
                //console.log(type(res.data))
                if (res.data == "-2") {
                    wx.showToast({
                        title: '数据未处理完',
                        icon: 'none',
                        duration: 5000
                    })
                    
                    return
                }
                that.setData({
                    level: res.data
                })
                wx.showToast({
                    title: '面瘫等级：' + res.data,
                    icon: 'success',
                    duration: 5000
                })
            },
            fail(err) {
                wx.showToast({
                    title: '获取失败！',
                    icon: 'none',
                    duration: 3000
                })
                console.log('获取失败！')
            },
            complete() {
                wx.hideLoading()
            }
        })

    },
    diagnose: function() {
        var that = this
        if (that.data.id == null) {
            wx.showToast({
                title: '请先获取您的唯一ID',
                icon: 'none',
                duration: 1000
            })
            return
        }
        if (that.data.img1 == null || that.data.img2 == null || that.data.video1 == null || that.data.video2 == null) {
            wx.showToast({
                title: '必须上传所有数据才能诊断',
                icon: 'none',
                duration: 1000
            })
            return
        }
        wx.showLoading({
            title: '后台算法处理',
        })
        console.log('后台算法开始处理')
        wx.request({
            url: ip + '/diag_and_return/' + that.data.id, //仅为示例，并非真实的接口地址
            data: {
                
            },
            header: {
                'content-type': 'application/json' // 默认值
            },
            success(res) {
                console.log(res.data)
                if (res.data == "-1") {
                    wx.showToast({
                        title: '数据处理过程出错，或者上传数据有问题（没有人像等），可过一段时间重试',
                        duration: 5000
                    })
                    that.setData({
                        level: '数据处理过程出错，或者上传数据有问题（没有人像等），可过一段时间重试'
                    })
                    return
                }
                that.setData({
                    level: res.data
                })
                wx.showToast({
                    title: '面瘫等级：' + res.data,
                    icon: 'success',
                    duration: 5000
                })
            },
            fail(err) {
                wx.showToast({
                    title: '获取失败！',
                    icon: 'none',
                    duration: 3000
                })
                console.log('诊断失败！')
            },
            complete() {
                wx.hideLoading()
            }
        })
        
    }
})