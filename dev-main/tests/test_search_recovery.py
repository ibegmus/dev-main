import os
import unittest
from unittest.mock import patch
from web_search import search_web

class SearchRecoveryTests(unittest.TestCase):
    @patch('web_search.DDGS')
    def test_fallback_after_primary_failure(self, factory):
        factory.return_value.text.side_effect=[TimeoutError('private detail'),[{'title':'Почвы','href':'https://example.org/soil','body':'Проверяемый фрагмент'}]]
        with patch.dict(os.environ,{'WEB_SEARCH_BACKEND':'bing,yandex'}):
            result=search_web('почвы')
        self.assertIsNone(result['web_error'])
        self.assertEqual(result['web_provider'],'yandex')
        self.assertEqual([r['status'] for r in result['web_attempts']],['failed','ok'])
        self.assertNotIn('private detail',str(result))

    @patch('web_search.DDGS')
    def test_exception_during_iteration_is_handled(self,factory):
        def broken():
            yield {'title':'partial','href':'https://example.org','body':'partial'}
            raise ConnectionError('secret-token')
        factory.return_value.text.side_effect=lambda *a,**kw:broken()
        with patch.dict(os.environ,{'WEB_SEARCH_BACKEND':'bing'}):result=search_web('почвы')
        self.assertIn('недоступен',result['web_error'])
        self.assertNotIn('secret-token',str(result))

    @patch('web_search.DDGS')
    def test_configured_proxy_and_ca_are_preserved(self,factory):
        factory.return_value.text.return_value=[]
        with patch.dict(os.environ,{'WEB_SEARCH_BACKEND':'bing','DDGS_PROXY':'http://configured-proxy:8080','SSL_CERT_FILE':'/custom/ca.pem'}):search_web('почвы')
        self.assertEqual(factory.call_args.kwargs['proxy'],'http://configured-proxy:8080')
        self.assertEqual(factory.call_args.kwargs['verify'],'/custom/ca.pem')

    @patch('web_search.DDGS')
    def test_standard_proxy_is_forwarded(self,factory):
        factory.return_value.text.return_value=[]
        with patch.dict(os.environ,{'WEB_SEARCH_BACKEND':'bing','DDGS_PROXY':'','HTTPS_PROXY':'http://session-proxy:8080'}):search_web('почвы')
        self.assertEqual(factory.call_args.kwargs['proxy'],'http://session-proxy:8080')

    @patch('web_search.DDGS')
    def test_empty_success_is_not_an_outage(self,factory):
        factory.return_value.text.side_effect=[TimeoutError(),[]]
        with patch.dict(os.environ,{'WEB_SEARCH_BACKEND':'bing,yandex'}):result=search_web('почвы')
        self.assertIsNone(result['web_error'])
        self.assertEqual(result['web_sources'],[])
