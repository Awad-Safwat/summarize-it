// ignore_for_file: prefer_const_constructors, avoid_print

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'summarized_text.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:http/http.dart' as http;

class TextPage extends StatefulWidget {
  const TextPage({Key? key}) : super(key: key);

  @override
  State<TextPage> createState() => _TextPageState();
}

class _TextPageState extends State<TextPage> {
  var mycontroller = TextEditingController();
  String? mytext;
  String? mysummary;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Your Text'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(5.0),
        child: Container(
          color: Colors.white,
          child: Column(
            mainAxisSize: MainAxisSize.max,
            children: [
              Expanded(
                child: TextField(
                  controller: mycontroller,
                  expands: true,
                  maxLines: null,
                  showCursor: true,
                  style:TextStyle(
                    fontSize: 23,
                    //fontWeight: FontWeight.bold,
                    
                  ) ,
                ),
              ),
              Container(
                width: double.infinity,
                color: Color.fromARGB(255, 202, 230, 255),
                child: TextButton(
                    onPressed: () async {
                      mytext = mycontroller.text;
                      EasyLoading.show(status: 'Summarizing...');
                      
                      var url =
                          Uri.http('10.0.2.2:5000', '/api', {'query': mytext});

                      var response = await http.get(url);
                      if (response.statusCode == 200) {
                        String resText =response.body;
                        EasyLoading.dismiss();
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) =>  SummarizedText(sumedText : resText)),
                        );
                      } else {
                        print(
                            'Request failed with status: ${response.statusCode}.');
                      }
                    },
                    child: Text(
                      'Summarize',
                      style: TextStyle(
                        fontSize: 25,
                        fontWeight: FontWeight.bold,
                      ),
                    )),
              )
            ],
          ),
        ),
      ),
    );
  }
}
