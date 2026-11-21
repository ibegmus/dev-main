import unittest
from soil_case import record_case
from soil_interview import validated_updates, reply, choose_question

class StructuredInterviewTests(unittest.TestCase):
    def test_quotes_must_be_in_current_message(self):
        self.assertEqual(validated_updates('Не знаю место',[
            {'field':'location','status':'known','quote':'Москва'},
            {'field':'pH','status':'known','quote':'Не знаю место'},
            {'field':'water','status':'unknown','quote':'место'},
            {'field':'location','status':'known','quote':'Не знаю место'}]),{})
        self.assertEqual(validated_updates('Свердловская область',[
            {'field':'location','status':'known','quote':'Свердловская область'}])['location']['status'],'known')

    def test_unknown_answer_is_bound_to_pending_question(self):
        first=record_case({},'Помогите определить почву')
        answer,first=reply(first,'Помогите определить почву')
        self.assertEqual(answer.count('?'),1)
        self.assertEqual(first['pending_field'],'location')
        second=record_case(first,'Не знаю',updates=[{'field':'water','status':'unknown','quote':'Не знаю'}])
        self.assertEqual(second['fields']['location']['status'],'unknown')
        self.assertNotIn('water',second['fields'])
        answer,second=reply(second,'Не знаю')
        self.assertNotIn('В каком регионе',answer)
        self.assertNotEqual(second['pending_field'],'location')

    def test_known_and_unknown_fields_survive_note_eviction(self):
        case=record_case({},'Свердловская область',updates=[{'field':'location','status':'known','quote':'Свердловская область'}])
        for i in range(12):case=record_case(case,f'Дополнительное наблюдение {i}')
        self.assertEqual(len(case['user_notes']),8)
        self.assertEqual(case['fields']['location']['quote'],'Свердловская область')
        text,case=reply(case,'Продолжим')
        self.assertNotIn('В каком регионе',text)

    def test_question_budget_and_no_repeated_questions(self):
        case={};asked=[]
        for i in range(5):
            case=record_case(case,'Не знаю' if i else 'Определи почву')
            text,case=reply(case,'Не знаю' if i else 'Определи почву','Светлый слой ниже тёмного')
            if case.get('pending_field'):asked.append(case['pending_field'])
        self.assertEqual(len(asked),3)
        self.assertEqual(len(set(asked)),3)
        self.assertNotIn('?',text)
        self.assertIn('Предварительный итог',text)

    def test_new_site_resets_observations_and_policy(self):
        case=record_case({},'Лес',updates=[{'field':'landscape','status':'known','quote':'Лес'}])
        _,case=reply(case,'Лес')
        case=record_case(case,'Новый участок',reset=True)
        self.assertEqual(case['fields'],{})
        self.assertEqual(case['asked_fields'],[])
        self.assertNotIn('pending_field',case)

    def test_finish_does_not_ask_or_guess_type(self):
        case=record_case({},'Подведите итог, больше данных нет')
        text,case=reply(case,'Подведите итог, больше данных нет','Белый слой и бурый горизонт')
        self.assertNotIn('?',text)
        self.assertNotIn('pending_field',case)
        self.assertIn('недостаточно',text)
        self.assertNotIn('серые лесные',text)

    def test_photo_changes_next_question_not_measured_fields(self):
        case={'fields':{'location':{'status':'unknown','quote':'Место неизвестно'}}}
        self.assertEqual(choose_question(case,'Белый слой'),'white_layer')
        self.assertEqual(choose_question(case,'Вода в разрезе'),'water')
        self.assertNotIn('water',case['fields'])

    def test_no_false_known_location_for_irrelevant_reply(self):
        case={'pending_field':'location','fields':{}}
        result=record_case(case,'я не понял')
        self.assertNotIn('location',result['fields'])

    def test_known_region_is_not_erased_by_unknown_city_in_same_message(self):
        rows=validated_updates('Свердловская область, город не знаю',[
            {'field':'location','status':'known','quote':'Свердловская область'},
            {'field':'location','status':'unknown','quote':'город не знаю'}])
        self.assertEqual(rows['location']['status'],'known')

    def test_legacy_case_keeps_notes_and_photos(self):
        old={'version':1,'user_notes':['Старое наблюдение'],'photos':['Старое фото']}
        new=record_case(old,'Новое наблюдение')
        self.assertEqual(new['photos'],old['photos'])
        self.assertEqual(new['user_notes'],['Старое наблюдение','Новое наблюдение'])
        self.assertEqual(new['fields'],{})

    def test_general_crop_question_does_not_start_identification(self):
        from soil_dialogue import identification_request
        self.assertFalse(identification_request('Какая почва лучше для моркови?'))
        self.assertFalse(identification_request('Что такое чернозём?'))
        self.assertTrue(identification_request('Определи почву на даче'))
        self.assertTrue(identification_request('Какая это почва на фото?'))

    def test_malformed_model_fields_do_not_break_the_request(self):
        self.assertEqual(validated_updates('Не знаю',[
            {'field':['location'],'status':'unknown','quote':'Не знаю'},
            {'field':'location','status':{},'quote':'Не знаю'},None]),{})
        case=record_case({'pending_field':{'invalid':True}},'Не знаю')
        self.assertEqual(case['fields'],{})

    def test_unrelated_capitalized_word_is_not_a_city(self):
        case=record_case({'pending_field':'location','fields':{}},'Сухо')
        self.assertNotIn('location',case['fields'])
