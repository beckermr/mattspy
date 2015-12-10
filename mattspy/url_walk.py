import os
import urllib2

class url_walk(object):
    """
    url_walk is just os.walk, but works for Apache file/directory lists online
    
    Example:
    
        for root,drs,fls in url_walk(webaddrs):
            # do something
            # os.path.join(root,fls) is the full web address to the files in fls
    
    """
    def __init__(self,base,user=None,password=None):
        self.base = base
        
        if user is not None and password is not None:
            password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
            top_level_url = base
            password_mgr.add_password(None, top_level_url, user, password)
            handler = urllib2.HTTPBasicAuthHandler(password_mgr)
            opener = urllib2.build_opener(handler)
            urllib2.install_opener(opener)
            
        self.queue = [base]

    def __iter__(self):
        while len(self.queue) > 0:
            root,dirs,fls = self._breakup_url(self.queue.pop(0))
            yield root,dirs,fls        
            for dr in dirs:
                self.queue.append(os.path.join(root,dr))        

    def _breakup_url(self,url):
        rep = urllib2.urlopen(url)
        html = rep.read()
        
        if url[-1] == '/':
            root = url[:-1]
        else:
            root = url
        dirs = []
        files = []
    
        for link in html.split('</a>')[:-1]:
            if 'alt="[DIR]"' not in link:
                is_file = True
            else:
                is_file = False
            items = link.split('<a ')[-1]
            items = items.split('>')
            tag = items[0].split('"')
            if len(tag)%2 != 0:
                tag = tag[:-1]
            props = {}
            for i in xrange(0,len(tag),2):
                props[str(tag[i][:-1].split())[2:-2]] = tag[i+1]
            nm = items[1]
            if nm not in ['Name','Last modified','Size','Description','Parent Directory']:
                if 'href' in props:
                    lapp = props['href']
                    if lapp[-1] == '/':
                        lapp = lapp[:-1]                    
                    if is_file:
                        files.append(lapp)
                    else:
                        dirs.append(lapp)

        return root,dirs,files

        
